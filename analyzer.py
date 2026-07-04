"""웹사이트 스크래핑 및 Ollama 분석 공통 로직.

CLI(analyze.py)와 Streamlit 앱(app.py)이 함께 사용하는 핵심 함수들을 모아둔다.
"""

import asyncio
import csv
import re
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import ollama

OLLAMA_MODEL = "qwen3.6"
OUTPUT_DIR = Path("output")
CRAWL_TIMEOUT_SECONDS = 60
PAGE_LOAD_WAIT_SECONDS = 2.0
MAX_ANALYSIS_CHARS = 20000

ANALYSIS_PROMPT = """다음은 웹사이트에서 스크래핑한 콘텐츠입니다. 이 사이트가 무엇을 하는 곳인지, \
주요 기능은 무엇인지, 기술 스택은 무엇인지 한국어로 명확한 분석 리포트를 작성하세요. \
되묻지 말고 바로 분석 결과만 출력하세요.

--- 스크래핑된 콘텐츠 ---
{content}
"""

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


async def scrape_site(url: str) -> str:
    log("스크래핑 중...")
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        page_timeout=CRAWL_TIMEOUT_SECONDS * 1000,
        wait_for="js:() => document.readyState === 'complete'",
        delay_before_return_html=PAGE_LOAD_WAIT_SECONDS,
    )

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=run_config),
                timeout=CRAWL_TIMEOUT_SECONDS,
            )
    except asyncio.TimeoutError:
        raise RuntimeError(f"스크래핑 타임아웃({CRAWL_TIMEOUT_SECONDS}초)이 발생했습니다: {url}")
    except Exception as exc:
        raise RuntimeError(f"사이트 접속에 실패했습니다: {url} ({exc})")

    if not result.success:
        raise RuntimeError(f"스크래핑에 실패했습니다: {result.error_message}")

    markdown = result.markdown.raw_markdown if hasattr(result.markdown, "raw_markdown") else str(result.markdown)
    if not markdown or not markdown.strip():
        raise RuntimeError("스크래핑된 콘텐츠가 비어 있습니다.")

    log(f"스크래핑 완료 ({len(markdown)}자)")
    return markdown


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


def analyze_content(content: str) -> str:
    log("Ollama로 분석 중...")

    truncated = content[:MAX_ANALYSIS_CHARS]
    analysis = _chat(ANALYSIS_PROMPT.format(content=truncated))
    if not analysis:
        raise RuntimeError(
            "Ollama가 빈 응답을 반환했습니다. "
            "(thinking 모드 모델이 답변을 생성하지 못했을 수 있습니다. 모델/num_predict 설정을 확인하세요.)"
        )

    log("분석 완료")
    return analysis


def save_outputs(raw_content: str, analysis: str) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / "raw_content.md"
    analysis_path = OUTPUT_DIR / "analysis.md"

    raw_path.write_text(raw_content, encoding="utf-8")
    analysis_path.write_text(analysis, encoding="utf-8")

    log(f"저장 완료: {raw_path}")
    log(f"저장 완료: {analysis_path}")

    return raw_path, analysis_path


def generate_output(content: str, instruction: str) -> str:
    log("Ollama로 생성 중...")

    truncated = content[:MAX_ANALYSIS_CHARS]
    prompt = CODE_GEN_PROMPT.format(content=truncated, instruction=instruction)
    result = _chat(prompt)
    if not result:
        raise RuntimeError(
            "Ollama가 빈 응답을 반환했습니다. "
            "(thinking 모드 모델이 답변을 생성하지 못했을 수 있습니다. 모델/num_predict 설정을 확인하세요.)"
        )

    log("생성 완료")
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
