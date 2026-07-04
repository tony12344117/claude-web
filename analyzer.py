"""웹사이트 스크래핑 및 Ollama 분석 공통 로직.

CLI(analyze.py)와 Streamlit 앱(app.py)이 함께 사용하는 핵심 함수들을 모아둔다.
정확도를 높이기 위해 두 단계 모두 자체 검증/검토 루프를 거친다:
- scrape_site(): 스크래핑 결과가 불완전(네비게이션/푸터만 있는 등)해 보이면 재시도.
  최대 MAX_SCRAPE_RETRIES회까지 시도하며, 완전해 보이면 조기 종료한다.
- generate_output(): 생성된 결과물을 MAX_REVIEW_ITERATIONS회 무조건 반복 검토·개선한다.
  모델이 "이제 됐다"고 판단해도 중간에 멈추지 않고 정해진 횟수를 전부 채운다.
"""

import asyncio
import csv
import re
from pathlib import Path
from typing import Callable

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import ollama

OLLAMA_MODEL = "qwen3.6"
OUTPUT_DIR = Path("output")
CRAWL_TIMEOUT_SECONDS = 60
PAGE_LOAD_WAIT_SECONDS = 2.0
MAX_ANALYSIS_CHARS = 20000

MAX_SCRAPE_RETRIES = 30
MIN_CONTENT_LENGTH = 200

MAX_REVIEW_ITERATIONS = 100

CODE_GEN_PROMPT = """당신은 웹 데이터 추출 및 코드 생성 전문가입니다.
아래는 웹사이트에서 스크래핑한 원본 콘텐츠와 사용자의 요청입니다.
사용자의 요청을 정확히 파악해서 그에 맞는 결과물만 생성하세요.

- 코드를 요청하면: 스크래핑된 사이트의 구조/로직/텍스트를 참고해서 실제로 작동하는 완전한 코드를 작성하세요. 설명은 최소화하고 코드 위주로 답하세요.
- 표/CSV를 요청하면: 정확한 CSV 형식으로 출력하세요.
- 요약을 요청하면: 핵심만 간결하게 정리하세요.

되묻지 말고 바로 결과물만 생성하세요.

[스크래핑 원본 콘텐츠]
{content}

[사용자 요청]
{instruction}
"""

REVIEW_PROMPT = """당신은 방금 아래 [이전 결과물]을 생성했습니다.
지금부터 [원본 콘텐츠]를 처음부터 끝까지 다시 꼼꼼히 읽고, [이전 결과물]의 모든 문장/코드/데이터 하나하나가
[원본 콘텐츠]의 실제 내용과 정확히 일치하는지 새로 대조 확인하세요. 이전에 확인했던 내용이라도
넘겨짚지 말고 [원본 콘텐츠]를 다시 근거로 삼아 재검증하세요.
[사용자 요청]에 정확히 부합하는지, 빠지거나 잘못되거나 지어낸(원본에 없는) 부분은 없는지 엄격하게 재검토하세요.

- 개선할 부분이 있다면: 개선된 최종 결과물 전체를 출력하고, 맨 첫 줄에 정확히 "REVISED"라고만 쓰세요.
- 더 이상 개선할 부분이 없다면: [이전 결과물]을 그대로 출력하고, 맨 첫 줄에 정확히 "FINAL"이라고만 쓰세요.
- 첫 줄 다음, 둘째 줄부터 결과물만 작성하세요. 설명, 인사말, 되묻는 말은 절대 하지 마세요.
- [이전 결과물]이 코드 블록(```언어\n...\n```)이나 CSV 형식이었다면, 개선된 결과물도 반드시 동일한 형식(같은 코드 펜스 언어 태그 등)을 그대로 유지하세요.

[원본 콘텐츠]
{content}

[사용자 요청]
{instruction}

[이전 결과물]
{previous_result}
"""

CODE_FENCE_PATTERN = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

LANGUAGE_EXTENSIONS = {
    "python": "py",
    "py": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "ts",
    "ts": "ts",
    "html": "html",
    "css": "css",
    "json": "json",
    "java": "java",
    "csharp": "cs",
    "cs": "cs",
    "go": "go",
    "ruby": "rb",
    "rb": "rb",
    "php": "php",
    "sql": "sql",
    "bash": "sh",
    "sh": "sh",
    "shell": "sh",
}


def log(message: str) -> None:
    print(f"[analyze] {message}", flush=True)


def _looks_incomplete(markdown: str) -> bool:
    text = markdown.strip()
    if len(text) < MIN_CONTENT_LENGTH:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True

    link_like = sum(1 for line in lines if line.startswith(("[", "*", "-")) and "](" in line)
    if len(lines) >= 5 and link_like / len(lines) > 0.8:
        return True  # 대부분 네비게이션/푸터 링크 목록으로만 구성된 경우

    return False


async def scrape_site(url: str, on_progress: Callable[[str], None] = log) -> str:
    last_markdown = ""
    last_error = None

    for attempt in range(1, MAX_SCRAPE_RETRIES + 1):
        on_progress(f"스크래핑 중... ({attempt}/{MAX_SCRAPE_RETRIES})")

        browser_config = BrowserConfig(headless=True)
        run_config = CrawlerRunConfig(
            page_timeout=CRAWL_TIMEOUT_SECONDS * 1000,
            wait_for="js:() => document.readyState === 'complete'",
            delay_before_return_html=PAGE_LOAD_WAIT_SECONDS + (attempt - 1) * 0.5,
        )

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=run_config),
                    timeout=CRAWL_TIMEOUT_SECONDS,
                )
        except asyncio.TimeoutError:
            last_error = f"스크래핑 타임아웃({CRAWL_TIMEOUT_SECONDS}초)이 발생했습니다: {url}"
            continue
        except Exception as exc:
            last_error = f"사이트 접속에 실패했습니다: {url} ({exc})"
            continue

        if not result.success:
            last_error = f"스크래핑에 실패했습니다: {result.error_message}"
            continue

        markdown = result.markdown.raw_markdown if hasattr(result.markdown, "raw_markdown") else str(result.markdown)
        if not markdown or not markdown.strip():
            last_error = "스크래핑된 콘텐츠가 비어 있습니다."
            continue

        last_markdown = markdown
        if not _looks_incomplete(markdown):
            on_progress(f"스크래핑 완료 ({len(markdown)}자, {attempt}번째 시도)")
            return markdown

    if last_markdown:
        on_progress(f"경고: {MAX_SCRAPE_RETRIES}번 재시도했지만 콘텐츠가 불완전할 수 있습니다. 마지막 결과를 사용합니다.")
        return last_markdown

    raise RuntimeError(last_error or "스크래핑에 실패했습니다.")


def _chat(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    try:
        try:
            # qwen3 계열처럼 hybrid thinking을 지원하는 모델은 think=False로 꺼야
            # 답변이 content가 아닌 thinking 필드로만 나오는 것을 막을 수 있다.
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages, think=False)
        except TypeError:
            # 설치된 ollama 라이브러리 버전이 think 파라미터를 지원하지 않는 경우.
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
    except Exception as exc:
        raise RuntimeError(f"Ollama 요청에 실패했습니다: {exc}")

    message = response.get("message", {})
    content = THINK_TAG_PATTERN.sub("", (message.get("content") or "")).strip()
    if not content:
        # think=False가 무시되는 모델의 경우, 답변이 thinking 필드에만 담겨 올 수 있다.
        content = (message.get("thinking") or "").strip()
    return content


def _refine_output(content: str, instruction: str, initial_result: str, on_progress: Callable[[str], None]) -> str:
    result = initial_result

    # 조기 종료 없이 무조건 MAX_REVIEW_ITERATIONS회를 전부 반복한다.
    # 모델이 "FINAL"(더 개선할 점 없음)이라고 답해도 멈추지 않고 다음 회차로 넘어간다.
    for i in range(1, MAX_REVIEW_ITERATIONS + 1):
        on_progress(f"결과 검토 중... ({i}/{MAX_REVIEW_ITERATIONS})")

        prompt = REVIEW_PROMPT.format(content=content, instruction=instruction, previous_result=result)
        response_text = _chat(prompt)
        if not response_text:
            continue  # 이번 회차 응답이 비어있으면 이전 결과를 유지한 채 다음 회차로

        first_line, _, rest = response_text.partition("\n")
        verdict = first_line.strip().upper()
        revised = rest.strip()

        if verdict.startswith(("REVISED", "FINAL")):
            if revised:
                result = revised
        else:
            # 모델이 형식을 지키지 않은 경우: 응답 전체를 개선된 결과물로 간주
            result = response_text.strip()

    on_progress(f"검토 {MAX_REVIEW_ITERATIONS}회 완료")
    return result


def generate_output(content: str, instruction: str, on_progress: Callable[[str], None] = log) -> str:
    on_progress("Ollama로 생성 중...")

    truncated = content[:MAX_ANALYSIS_CHARS]
    prompt = CODE_GEN_PROMPT.format(content=truncated, instruction=instruction)
    result = _chat(prompt)
    if not result:
        raise RuntimeError(
            "Ollama가 빈 응답을 반환했습니다. "
            "(thinking 모드 모델이 답변을 생성하지 못했을 수 있습니다. 모델/num_predict 설정을 확인하세요.)"
        )

    result = _refine_output(truncated, instruction, result, on_progress)

    on_progress("생성 완료")
    return result


def _looks_like_csv(text: str) -> bool:
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    try:
        csv.Sniffer().sniff(text[:2000], delimiters=",")
    except csv.Error:
        return False
    return True


def save_result(response: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    match = CODE_FENCE_PATTERN.search(response)
    if match:
        lang = match.group(1).strip().lower()
        code = match.group(2).strip()
        ext = LANGUAGE_EXTENSIONS.get(lang, "txt")
        path = OUTPUT_DIR / f"result.{ext}"
        path.write_text(code + "\n", encoding="utf-8")
        log(f"저장 완료: {path}")
        return path

    if _looks_like_csv(response):
        path = OUTPUT_DIR / "result.csv"
        path.write_text(response.strip() + "\n", encoding="utf-8")
        log(f"저장 완료: {path}")
        return path

    path = OUTPUT_DIR / "result.md"
    path.write_text(response.strip() + "\n", encoding="utf-8")
    log(f"저장 완료: {path}")
    return path
