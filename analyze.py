#!/usr/bin/env python3
"""웹사이트 분석기: Crawl4AI로 스크래핑하고 Ollama(qwen3.6)로 분석한다.

사용법:
    python analyze.py <URL>
"""

import asyncio
import sys
from pathlib import Path

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import ollama

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

OLLAMA_MODEL = "qwen3.6"
OUTPUT_DIR = Path("output")
CRAWL_TIMEOUT_SECONDS = 60
OLLAMA_TIMEOUT_SECONDS = 300
PAGE_LOAD_WAIT_SECONDS = 2

ANALYSIS_PROMPT = """다음은 웹사이트에서 스크래핑한 콘텐츠입니다. 이 사이트가 무엇을 하는 곳인지, \
주요 기능은 무엇인지, 기술적으로 어떤 스택을 쓰는지 한국어로 명확하게 분석 리포트를 작성하세요. \
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

    max_chars = 20000
    truncated = content[:max_chars]

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


def save_outputs(raw_content: str, analysis: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = OUTPUT_DIR / "raw_content.md"
    analysis_path = OUTPUT_DIR / "analysis.md"

    raw_path.write_text(raw_content, encoding="utf-8")
    analysis_path.write_text(analysis, encoding="utf-8")

    log(f"저장 완료: {raw_path}")
    log(f"저장 완료: {analysis_path}")


async def run(url: str) -> None:
    raw_content = await scrape_site(url)
    analysis = analyze_content(raw_content)
    save_outputs(raw_content, analysis)
    log("모든 작업이 완료되었습니다.")


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python analyze.py <URL>", file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1]
    if not url.startswith(("http://", "https://")):
        print(f"오류: 유효한 URL이 아닙니다: {url}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run(url))
    except RuntimeError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
