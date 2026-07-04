#!/usr/bin/env python3
"""웹사이트 분석기 CLI: Crawl4AI로 스크래핑하고 Ollama(qwen3.6)로 분석한다.

사용법:
    python analyze.py <URL>
"""

import asyncio
import sys

from analyzer import scrape_site, analyze_content, save_outputs, log

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


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
