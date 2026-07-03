#!/usr/bin/env python3
"""웹사이트 분석기: URL을 입력받아 브라우저 렌더링, 기술 스택/구조 추출,
Claude API 분석을 거쳐 결과물을 ZIP으로 묶어 출력한다.

사용법:
    python analyze.py <URL>
"""

from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from dotenv import load_dotenv
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

TIMEOUT_MS = 30_000
TIMEOUT_S = 30.0
CLAUDE_MODEL = "claude-sonnet-4-6"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 WebsiteAnalyzerBot/1.0"
)

EXTRACT_JS = """
() => {
  const metas = Array.from(document.querySelectorAll('meta')).map(m => ({
    name: m.getAttribute('name'),
    property: m.getAttribute('property'),
    content: m.getAttribute('content'),
  }));
  const scripts = Array.from(document.querySelectorAll('script[src]')).map(s => s.src);
  const inlineScriptSnippets = Array.from(document.querySelectorAll('script:not([src])'))
    .map(s => (s.textContent || '').slice(0, 200));
  const stylesheets = Array.from(document.querySelectorAll('link[rel="stylesheet"]')).map(l => l.href);
  const links = Array.from(document.querySelectorAll('a[href]')).map(a => a.href);
  const htmlAttrs = {};
  for (const attr of document.documentElement.attributes) {
    htmlAttrs[attr.name] = attr.value;
  }
  const generatorRoots = ['__NEXT_DATA__', '__NUXT__', '__remixContext', '__gatsby'];
  const globalMarkers = generatorRoots.filter(k => typeof window[k] !== 'undefined');
  return {
    title: document.title,
    metas,
    scripts,
    inlineScriptSnippets,
    stylesheets,
    links,
    htmlAttrs,
    globalMarkers,
    bodyText: document.body ? document.body.innerText : '',
  };
}
"""


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def normalize_url(raw_url: str) -> str:
    if not re.match(r"^https?://", raw_url, re.IGNORECASE):
        raw_url = "https://" + raw_url
    return raw_url


def get_domain(url: str) -> str:
    netloc = urlparse(url).netloc
    return netloc.split(":")[0].removeprefix("www.")


def check_robots_allowed(url: str) -> bool:
    """robots.txt를 확인해 해당 경로 접근이 허용되는지 반환. robots.txt가 없거나
    가져오지 못하면 관례적으로 허용으로 간주한다."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    try:
        resp = httpx.get(
            robots_url,
            timeout=TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        if resp.status_code >= 400:
            return True
        parser.parse(resp.text.splitlines())
    except httpx.HTTPError as exc:
        eprint(f"[경고] robots.txt 조회 실패 ({exc}) - 접근 허용으로 간주합니다.")
        return True
    return parser.can_fetch(USER_AGENT, url)


def _find_bundled_chromium() -> str | None:
    """PLAYWRIGHT_BROWSERS_PATH에 미리 설치된 chromium 실행 파일을 직접 찾는다.
    playwright 패키지 버전과 사전 설치된 브라우저 리비전이 어긋나 자동 탐색이
    실패하는 경우(예: 컨테이너에 브라우저가 미리 구워져 있는 환경)의 폴백이다."""
    import glob

    browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if not browsers_path:
        return None
    candidates = sorted(glob.glob(os.path.join(browsers_path, "chromium-*/chrome-linux/chrome")))
    return candidates[-1] if candidates else None


def render_site(url: str) -> dict[str, Any]:
    """Playwright(Chromium)로 실제 브라우저처럼 접속해 렌더링된 HTML/스크린샷/구조를 수집."""
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except PlaywrightError:
            fallback_path = _find_bundled_chromium()
            if not fallback_path:
                raise
            browser = p.chromium.launch(executable_path=fallback_path)
        try:
            context = browser.new_context(user_agent=USER_AGENT)
            page = context.new_page()
            page.set_default_timeout(TIMEOUT_MS)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            except PlaywrightTimeoutError as exc:
                raise RuntimeError(f"페이지 접속 타임아웃({TIMEOUT_MS}ms): {url}") from exc

            # SPA의 JS 렌더링을 기다리되, 끝까지 idle이 되지 않아도 치명적 오류로 취급하지 않는다.
            try:
                page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
            except PlaywrightTimeoutError:
                eprint("[경고] networkidle 대기 시간 초과 - 현재까지 렌더링된 내용으로 계속 진행합니다.")

            html = page.content()
            extracted = page.evaluate(EXTRACT_JS)
            screenshot_bytes = page.screenshot(full_page=True)

            return {
                "html": html,
                "screenshot": screenshot_bytes,
                **extracted,
            }
        finally:
            browser.close()


def detect_tech_stack(data: dict[str, Any], html: str) -> dict[str, list[str]]:
    """메타태그, 스크립트 src, 전역 변수 흔적 등으로 기술 스택을 추론."""
    scripts = " ".join(data.get("scripts", []))
    stylesheets = " ".join(data.get("stylesheets", []))
    metas = data.get("metas", [])
    global_markers = set(data.get("globalMarkers", []))
    html_attrs = data.get("htmlAttrs", {})
    inline_snippets = " ".join(data.get("inlineScriptSnippets", []))
    haystack = " ".join([scripts, stylesheets, inline_snippets, html]).lower()

    generator = ""
    for m in metas:
        if (m.get("name") or "").lower() == "generator":
            generator = (m.get("content") or "").lower()
            break

    frameworks: list[str] = []
    if "__next_data__" in global_markers or "_next/static" in haystack or "next.js" in generator:
        frameworks.append("Next.js")
    if "__nuxt__" in global_markers or "_nuxt/" in haystack:
        frameworks.append("Nuxt.js")
    if "__remixcontext" in global_markers:
        frameworks.append("Remix")
    if "__gatsby" in global_markers or "gatsby" in haystack:
        frameworks.append("Gatsby")
    if "data-reactroot" in html.lower() or "react" in haystack:
        frameworks.append("React")
    if "ng-version" in html_attrs or "angular" in haystack:
        frameworks.append("Angular")
    if re.search(r"vue(\.min)?\.js|__vue__|data-v-", haystack):
        frameworks.append("Vue.js")
    if "svelte" in haystack:
        frameworks.append("Svelte")
    if "jquery" in haystack:
        frameworks.append("jQuery")
    if "wordpress" in generator or "wp-content" in haystack or "wp-includes" in haystack:
        frameworks.append("WordPress")
    if "shopify" in haystack or "cdn.shopify.com" in haystack:
        frameworks.append("Shopify")
    if "wix.com" in haystack or "wixstatic" in haystack:
        frameworks.append("Wix")
    if "squarespace" in haystack:
        frameworks.append("Squarespace")
    if "webflow" in haystack:
        frameworks.append("Webflow")

    css_frameworks: list[str] = []
    if "tailwind" in haystack:
        css_frameworks.append("Tailwind CSS")
    if "bootstrap" in haystack:
        css_frameworks.append("Bootstrap")
    if "bulma" in haystack:
        css_frameworks.append("Bulma")
    if "materialize" in haystack:
        css_frameworks.append("Materialize")

    analytics_and_infra: list[str] = []
    if "googletagmanager.com" in haystack or "gtag(" in haystack:
        analytics_and_infra.append("Google Analytics / GTM")
    if "cloudflare" in haystack:
        analytics_and_infra.append("Cloudflare")
    if "hotjar" in haystack:
        analytics_and_infra.append("Hotjar")
    if "sentry" in haystack:
        analytics_and_infra.append("Sentry")
    if "stripe.com" in haystack or "stripe.js" in haystack:
        analytics_and_infra.append("Stripe")
    if "vercel" in haystack:
        analytics_and_infra.append("Vercel")

    return {
        "frameworks": sorted(set(frameworks)),
        "css_frameworks": sorted(set(css_frameworks)),
        "analytics_and_infra": sorted(set(analytics_and_infra)),
        "generator_meta": generator,
    }


def extract_internal_links(links: list[str], base_url: str) -> list[str]:
    domain = get_domain(base_url)
    internal: set[str] = set()
    for href in links:
        try:
            resolved = urljoin(base_url, href)
            parsed = urlparse(resolved)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https"):
            continue
        if get_domain(resolved) != domain:
            continue
        cleaned = parsed._replace(fragment="").geturl()
        internal.add(cleaned)
    return sorted(internal)


def build_metadata(url: str, data: dict[str, Any], tech_stack: dict[str, list[str]]) -> dict[str, Any]:
    metas = data.get("metas", [])

    def meta_content(*, name: str | None = None, prop: str | None = None) -> str:
        for m in metas:
            if name and (m.get("name") or "").lower() == name.lower():
                return m.get("content") or ""
            if prop and (m.get("property") or "").lower() == prop.lower():
                return m.get("content") or ""
        return ""

    description = meta_content(name="description")
    og_description = meta_content(prop="og:description")
    og_title = meta_content(prop="og:title")
    og_site_name = meta_content(prop="og:site_name")

    internal_links = extract_internal_links(data.get("links", []), url)
    body_text = data.get("bodyText", "") or ""

    return {
        "url": url,
        "domain": get_domain(url),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "title": data.get("title", ""),
        "description": description,
        "og_title": og_title,
        "og_description": og_description,
        "og_site_name": og_site_name,
        "tech_stack": tech_stack,
        "internal_links": internal_links,
        "internal_link_count": len(internal_links),
        "external_script_count": len(data.get("scripts", [])),
        "stylesheet_count": len(data.get("stylesheets", [])),
        "body_text_excerpt": body_text[:3000],
    }


def call_claude_analysis(metadata: dict[str, Any], body_text: str, html_len: int) -> str:
    """Claude API로 사이트 목적/기술스택/주요 기능을 한국어로 분석."""
    from anthropic import Anthropic

    client = Anthropic()

    tech = metadata["tech_stack"]
    tech_summary = (
        f"- 프레임워크/플랫폼: {', '.join(tech['frameworks']) or '탐지되지 않음'}\n"
        f"- CSS 프레임워크: {', '.join(tech['css_frameworks']) or '탐지되지 않음'}\n"
        f"- 분석/인프라: {', '.join(tech['analytics_and_infra']) or '탐지되지 않음'}\n"
        f"- generator 메타태그: {tech['generator_meta'] or '없음'}"
    )

    links_sample = "\n".join(f"- {link}" for link in metadata["internal_links"][:40])

    prompt = f"""다음은 웹사이트를 자동 수집한 정보입니다. 이 정보를 바탕으로 한국어로 분석 리포트를 작성해주세요.

## 수집된 정보

URL: {metadata['url']}
페이지 제목: {metadata['title']}
meta description: {metadata['description'] or '없음'}
og:description: {metadata['og_description'] or '없음'}
og:title: {metadata['og_title'] or '없음'}
og:site_name: {metadata['og_site_name'] or '없음'}
HTML 전체 길이: {html_len:,}자
내부 링크 수: {metadata['internal_link_count']}

추론된 기술 스택:
{tech_summary}

내부 링크 샘플 (최대 40개):
{links_sample or '없음'}

본문 텍스트 발췌 (최대 3000자):
{body_text[:3000]}

## 요청 사항

아래 구조의 마크다운 리포트를 작성하세요:

# {metadata['domain']} 분석 리포트

## 1. 사이트 개요
이 사이트가 무엇을 하는 곳인지, 누구를 위한 서비스인지 2~4문단으로 한국어로 설명.

## 2. 기술 스택
탐지된 기술 스택을 정리하고, 왜 그렇게 판단했는지 근거를 간단히 덧붙임.

## 3. 주요 기능 및 섹션
내부 링크 구조와 본문 내용을 바탕으로 주요 기능/섹션을 목록으로 정리.

## 4. 종합 의견
사이트의 특징, 타겟 사용자, 개선 여지 등에 대한 간단한 종합 의견.

내용이 부족한 항목은 "정보가 충분하지 않습니다"라고 명시하고 추측을 단정적으로 서술하지 마세요."""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_parts).strip()


def build_zip(
    output_path: str,
    html: str,
    screenshot_bytes: bytes,
    analysis_md: str,
    metadata: dict[str, Any],
) -> None:
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", html)
        zf.writestr("screenshot.png", screenshot_bytes)
        zf.writestr("analysis.md", analysis_md)
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))


def main() -> None:
    if len(sys.argv) != 2:
        eprint("사용법: python analyze.py <URL>")
        sys.exit(1)

    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        eprint("[오류] ANTHROPIC_API_KEY 환경변수가 설정되어 있지 않습니다 (.env 파일 확인).")
        sys.exit(1)

    url = normalize_url(sys.argv[1].strip())

    parsed = urlparse(url)
    if not parsed.netloc:
        eprint(f"[오류] 올바르지 않은 URL입니다: {sys.argv[1]}")
        sys.exit(1)

    print(f"[1/5] robots.txt 확인 중: {url}")
    try:
        if not check_robots_allowed(url):
            eprint(f"[오류] robots.txt에서 해당 경로 접근을 허용하지 않습니다: {url}")
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001 - robots 조회 자체의 예상치 못한 오류
        eprint(f"[경고] robots.txt 확인 중 오류 발생 ({exc}) - 접근 허용으로 간주합니다.")

    print("[2/5] 브라우저로 접속 및 렌더링 중 (JS 렌더링 포함)...")
    try:
        rendered = render_site(url)
    except RuntimeError as exc:
        eprint(f"[오류] {exc}")
        sys.exit(1)
    except PlaywrightError as exc:
        eprint(f"[오류] 브라우저 접속 실패: {exc}")
        sys.exit(1)

    print("[3/5] 기술 스택 및 구조 데이터 추출 중...")
    try:
        tech_stack = detect_tech_stack(rendered, rendered["html"])
        metadata = build_metadata(url, rendered, tech_stack)
    except Exception as exc:  # noqa: BLE001
        eprint(f"[오류] 데이터 추출 중 문제가 발생했습니다: {exc}")
        sys.exit(1)

    print(f"[4/5] Claude API({CLAUDE_MODEL})로 사이트 분석 중...")
    try:
        analysis_md = call_claude_analysis(metadata, rendered.get("bodyText", ""), len(rendered["html"]))
    except Exception as exc:  # noqa: BLE001 - anthropic SDK 예외 포함
        eprint(f"[오류] Claude API 분석 실패: {exc}")
        sys.exit(1)

    domain = metadata["domain"]
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = f"{domain}_{date_str}.zip"

    print(f"[5/5] 결과물을 {output_path} 로 압축 중...")
    try:
        build_zip(output_path, rendered["html"], rendered["screenshot"], analysis_md, metadata)
    except OSError as exc:
        eprint(f"[오류] ZIP 파일 생성 실패: {exc}")
        sys.exit(1)

    print(f"완료: {output_path}")


if __name__ == "__main__":
    main()
