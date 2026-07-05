"""웹사이트 스크래핑 및 Ollama 분석 공통 로직.

CLI(analyze.py)와 Streamlit 앱(app.py)이 함께 사용하는 핵심 함수들을 모아둔다.
정확도를 높이기 위해 두 단계 모두 자체 검증 루프를 거친다:
- scrape_site(): 스크래핑 결과가 불완전(네비게이션/푸터만 있는 등)해 보이면 재시도.
  최대 MAX_SCRAPE_RETRIES회까지 시도하며, 완전해 보이면 조기 종료한다.
- generate_output(): 생성된 결과물을 검증(버그/잘못된 데이터/누락된 데이터 확인)하고,
  문제가 있으면 하나씩 고쳐서 다시 검증한다. CLEAN_STREAK_REQUIRED회 연속으로
  문제가 없다고 확인되면 그 결과를 출력하고, 그렇지 못하면 MAX_REVIEW_ITERATIONS회까지
  계속 검증·수정을 반복한다.
  매번 원본 콘텐츠를 새 프롬프트에 다시 붙여넣는 대신, Ollama와의 대화(messages)를
  하나로 유지해서 같은 맥락(기억)이 남은 상태에서 후속 질문만 이어서 묻는다.
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
CLEAN_STREAK_REQUIRED = 3

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

VERIFY_PROMPT = """방금 당신이 만든 결과물을 처음부터 다시 엄격하게 검증하세요.
이전에 확인했던 내용이라도 넘겨짚지 말고, 대화 맨 처음에 주어졌던 원본 콘텐츠와 사용자 요청을
처음부터 다시 읽고 대조하세요.

다음 세 가지를 모두 확인하세요:
1. 버그나 오류가 있는가? (코드라면 실행되지 않거나 문법이 틀린 부분, CSV라면 형식이 깨진 부분 등)
2. 사실과 다르거나 지어낸(원본에 없는) 데이터가 있는가?
3. 사용자 요청을 처리하는 데 필요한 데이터를 원본 콘텐츠에서 놓치거나 빠뜨리지는 않았는가?

- 문제가 하나도 없다면: 정확히 "NONE"이라고만 답하세요.
- 문제가 하나라도 있다면: 첫 줄에 정확히 "ISSUE"라고 쓰고, 둘째 줄부터 무엇이 문제인지 구체적으로 설명하세요.
- 그 외의 설명, 인사말, 되묻는 말은 절대 하지 마세요.
"""

FIX_PROMPT = """방금 지적한 아래 문제를 해결하세요.

[발견된 문제]
{issue}

대화 맨 처음에 주어졌던 원본 콘텐츠와 사용자 요청을 다시 참고해서, 이 문제를 해결한 결과물
전체를 처음부터 다시 작성하세요. 방금 전 결과물이 코드 블록(```언어\n...\n```)이나 CSV
형식이었다면, 수정된 결과물도 반드시 동일한 형식을 그대로 유지하세요.
설명, 인사말, 되묻는 말 없이 수정된 결과물만 출력하세요.
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


def _chat_turn(messages: list[dict]) -> str:
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


def _ask(messages: list[dict], prompt: str) -> str:
    """messages에 새 user 턴을 추가해 같은 대화 맥락 안에서 다시 묻고, 응답을 대화 이력에 반영한다."""
    messages.append({"role": "user", "content": prompt})
    reply = _chat_turn(messages)
    if reply:
        messages.append({"role": "assistant", "content": reply})
    else:
        # 빈 응답이면 이번 user 턴은 대화 이력에서 되돌려 다음 시도에 영향을 주지 않게 한다.
        messages.pop()
    return reply


def _verify(messages: list[dict]) -> str | None:
    """결과물을 검증한다. 문제가 없으면 None, 응답을 못 받으면 '', 문제가 있으면 그 설명을 반환."""
    response = _ask(messages, VERIFY_PROMPT)
    if not response:
        return ""
    if response.strip().upper().startswith("NONE"):
        return None
    return response.strip()


def _fix(messages: list[dict], issue: str) -> str:
    return _ask(messages, FIX_PROMPT.format(issue=issue))


def _verify_and_fix(messages: list[dict], initial_result: str, on_progress: Callable[[str], None]) -> str:
    result = initial_result
    clean_streak = 0

    for attempt in range(1, MAX_REVIEW_ITERATIONS + 1):
        on_progress(f"검증 중... ({clean_streak}/{CLEAN_STREAK_REQUIRED} 연속 통과, 총 {attempt}/{MAX_REVIEW_ITERATIONS}회)")
        issue = _verify(messages)

        if issue is None:
            clean_streak += 1
            if clean_streak >= CLEAN_STREAK_REQUIRED:
                on_progress(f"검증 완료: {CLEAN_STREAK_REQUIRED}회 연속 문제 없음 확인됨")
                return result
            continue

        if issue == "":
            continue  # 검증 응답을 못 받음: 판정 보류하고 스트릭 유지한 채 다음 시도로

        clean_streak = 0
        on_progress(f"문제 발견, 수정 중... ({attempt}/{MAX_REVIEW_ITERATIONS})")
        fixed = _fix(messages, issue)
        if fixed:
            result = fixed

    on_progress(f"경고: 최대 {MAX_REVIEW_ITERATIONS}회까지 시도했지만 완전히 해결하지 못했습니다. 마지막 결과를 사용합니다.")
    return result


def generate_output(content: str, instruction: str, on_progress: Callable[[str], None] = log) -> str:
    on_progress("Ollama로 생성 중...")

    truncated = content[:MAX_ANALYSIS_CHARS]
    initial_prompt = CODE_GEN_PROMPT.format(content=truncated, instruction=instruction)
    messages = [{"role": "user", "content": initial_prompt}]
    result = _chat_turn(messages)
    if not result:
        raise RuntimeError(
            "Ollama가 빈 응답을 반환했습니다. "
            "(thinking 모드 모델이 답변을 생성하지 못했을 수 있습니다. 모델/num_predict 설정을 확인하세요.)"
        )
    messages.append({"role": "assistant", "content": result})

    result = _verify_and_fix(messages, result, on_progress)

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
