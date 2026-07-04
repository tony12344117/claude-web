#!/usr/bin/env python3
"""웹사이트 데이터 추출 및 코드 생성 CLI.

URL과 자연어 지시문을 받아 Crawl4AI로 스크래핑한 뒤, Ollama(qwen3.6)에게
지시문에 맞는 결과물(코드/CSV/요약 등)을 생성시키고 output/ 폴더에 저장한다.

사용법:
    python analyze.py <URL> "<지시문>"

예시:
    python analyze.py https://costmyhome.com "이 사이트처럼 작동하는 견적 계산기 코드 만들어줘"
    python analyze.py https://example.com "상품명이랑 가격만 CSV로 뽑아줘"
    python analyze.py https://news.com "핵심 내용 3줄 요약해줘"
"""

import argparse
import asyncio
import sys

from analyzer import scrape_site, generate_output, save_result, log

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


async def run(url: str, instruction: str) -> None:
    raw_content = await scrape_site(url)
    result = generate_output(raw_content, instruction)
    path = save_result(result)
    log("모든 작업이 완료되었습니다.")
    print()
    print(result)
    print()
    log(f"결과 파일: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="URL을 스크래핑하고 자연어 지시에 따라 결과물(코드/CSV/요약 등)을 생성합니다.",
    )
    parser.add_argument("url", help="분석할 웹사이트 URL")
    parser.add_argument("instruction", help="Ollama에게 전달할 자연어 지시문")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.url.startswith(("http://", "https://")):
        print(f"오류: 유효한 URL이 아닙니다: {args.url}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(run(args.url, args.instruction))
    except RuntimeError as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
