"""웹사이트 스크래핑 및 Ollama 분석 공통 로직.

CLI(analyze.py)와 Streamlit 앱(app.py)이 함께 사용하는 핵심 함수들을 모아둔다.
"""

import asyncio
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import ollama

OLLAMA_MODEL = "qwen3.6"
OUTPUT_DIR = Path("output")
CRAWL_TIMEOUT_SECONDS = 60
PAGE_LOAD_WAIT_SECONDS = 2
MAX_ANALYSIS_CHARS = 20000

ANALYSIS_PROMPT = """다음은 웹사이트에서 스크래핑한 콘텐츠입니다. 이 사이트가 무엇을 하는 곳인지, \
주요 기능은 무엇인지, 기술 스택은 무엇인지 한국어로 명확한 분석 리포트를 작성하세요. \
되묻지 말고 바로 분석 결과만 출력하세요.

--- 스크래핑된 콘텐츠 ---
{content}
"""


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


def analyze_content(content: str) -> str:
    log("Ollama로 분석 중...")

    truncated = content[:MAX_ANALYSIS_CHARS]

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": ANALYSIS_PROMPT.format(content=truncated)}],
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama 분석에 실패했습니다: {exc}")

    analysis = response.get("message", {}).get("content", "").strip()
    if not analysis:
        raise RuntimeError("Ollama가 빈 응답을 반환했습니다.")

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
