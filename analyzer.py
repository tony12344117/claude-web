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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Union

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
import ollama


@dataclass
class ScrapedData:
    """스크래핑으로 수집한 전체 데이터. Ollama가 이 중 필요한 것만 골라서 사용한다."""

    markdown: str
    html: str = ""
    network_requests: list = field(default_factory=list)

OLLAMA_MODEL = "qwen3.6"
OUTPUT_DIR = Path("output")
CRAWL_TIMEOUT_SECONDS = 60
PAGE_LOAD_WAIT_SECONDS = 2.0
MAX_ANALYSIS_CHARS = 20000

MAX_SCRAPE_RETRIES = 30
MIN_CONTENT_LENGTH = 200

MAX_REVIEW_ITERATIONS = 100
CLEAN_STREAK_REQUIRED = 3

# 모든 데이터는 사용자가 이 도구로 직접 스크래핑해서 수집한 것이고, 요청 자체도
# 사용자가 이미 승인한 작업이라는 점을 모델에게 명시해서, 저작권/스크래핑 정책 같은
# 일반론적인 이유로 되묻거나 거절하지 말고 바로 결과물을 생성하도록 안내한다.
# (실제로 위법하거나 위험한 내용을 요청한 경우까지 강제로 생성하라는 지시는 아니다.)
COMPLIANCE_NOTE = """참고: 이 콘텐츠는 사용자가 자신의 도구로 직접 스크래핑해서 수집한 데이터이고,
아래 요청은 사용자가 이미 승인한 정당한 작업입니다. 저작권, 이용약관, 스크래핑 정책 등을
근거로 임의로 되묻거나 거절하지 말고, 요청받은 결과물을 바로 생성하세요.
(단, 요청 내용 자체가 실제로 불법적이거나 타인에게 위해를 가하는 목적이 명백한 경우는 예외입니다.)
"""

SELECT_DATA_PROMPT = """당신은 웹 데이터 추출 및 코드 생성 전문가입니다.
웹사이트를 스크래핑해서 아래 세 종류의 데이터를 수집해 두었습니다.
사용자 요청을 처리하는 데 꼭 필요한 데이터만 고르세요.

1. MARKDOWN: 페이지 본문을 마크다운으로 변환한 텍스트 ({markdown_chars}자)
2. HTML: 페이지 원본 HTML 소스 — 태그 구조, 클래스명, 스크립트 등 포함 ({html_chars}자)
3. NETWORK: 페이지 로딩 중 발생한 HTTP 네트워크 요청 정보 — URL, 메서드, 헤더, 상태코드 ({network_count}건)

[네트워크 요청 미리보기]
{network_preview}

[사용자 요청]
{instruction}

필요한 데이터 이름만 쉼표로 구분해서 한 줄로 답하세요. (예: MARKDOWN 또는 MARKDOWN,NETWORK 또는 HTML)
다른 설명은 절대 하지 마세요.
"""

CODE_GEN_PROMPT = """좋습니다. 요청하신 데이터는 아래와 같습니다.
사용자의 요청을 정확히 파악해서 그에 맞는 결과물만 생성하세요.

- 코드를 요청하면: 스크래핑된 사이트의 구조/로직/텍스트를 참고해서 실제로 작동하는 완전한 코드를 작성하세요. 설명은 최소화하고 코드 위주로 답하세요.
- 표/CSV를 요청하면: 정확한 CSV 형식으로 출력하세요.
- 요약을 요청하면: 핵심만 간결하게 정리하세요.

되묻지 말고 바로 결과물만 생성하세요.

{compliance_note}
{data_sections}

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

{compliance_note}
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


async def scrape_site(url: str, on_progress: Callable[[str], None] = log) -> ScrapedData:
    last_data = None
    last_error = None

    for attempt in range(1, MAX_SCRAPE_RETRIES + 1):
        on_progress(f"스크래핑 중... ({attempt}/{MAX_SCRAPE_RETRIES})")

        browser_config = BrowserConfig(headless=True)
        run_config = CrawlerRunConfig(
            page_timeout=CRAWL_TIMEOUT_SECONDS * 1000,
            wait_for="js:() => document.readyState === 'complete'",
            delay_before_return_html=PAGE_LOAD_WAIT_SECONDS + (attempt - 1) * 0.5,
            capture_network_requests=True,
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

        data = ScrapedData(
            markdown=markdown,
            html=result.html or "",
            network_requests=result.network_requests or [],
        )

        last_data = data
        if not _looks_incomplete(markdown):
            on_progress(
                f"스크래핑 완료 (본문 {len(markdown)}자, HTML {len(data.html)}자, "
                f"네트워크 요청 {len(data.network_requests)}건, {attempt}번째 시도)"
            )
            return data

    if last_data:
        on_progress(f"경고: {MAX_SCRAPE_RETRIES}번 재시도했지만 콘텐츠가 불완전할 수 있습니다. 마지막 결과를 사용합니다.")
        return last_data

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
    return _ask(messages, FIX_PROMPT.format(issue=issue, compliance_note=COMPLIANCE_NOTE))


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


def _format_network_requests(requests: list, max_entries: int = 100, with_headers: bool = True) -> str:
    """캡처된 네트워크 요청을 'METHOD URL / 상태 / 헤더' 형태의 텍스트로 정리한다."""
    lines = []
    for event in requests[:max_entries]:
        if not isinstance(event, dict):
            lines.append(str(event))
            continue

        event_type = event.get("event_type", "")
        method = event.get("method", "")
        req_url = event.get("url", "")
        status = event.get("status", "")

        if event_type == "response":
            lines.append(f"← {status} {req_url}")
        else:
            lines.append(f"→ {method or 'GET'} {req_url}")

        if with_headers:
            headers = event.get("headers") or {}
            if isinstance(headers, dict) and headers:
                for key, value in list(headers.items())[:10]:
                    lines.append(f"    {key}: {str(value)[:200]}")

    if len(requests) > max_entries:
        lines.append(f"... (외 {len(requests) - max_entries}건 생략)")
    return "\n".join(lines)


def _parse_data_selection(reply: str) -> set[str]:
    """데이터 선택 응답에서 MARKDOWN/HTML/NETWORK 키워드를 추출한다. 없으면 MARKDOWN 기본값."""
    upper = reply.upper()
    selected = {name for name in ("MARKDOWN", "HTML", "NETWORK") if name in upper}
    return selected or {"MARKDOWN"}


def _build_data_sections(scraped: ScrapedData, selected: set[str]) -> str:
    """선택된 데이터 소스만 모아 프롬프트에 넣을 본문을 만든다. 전체 분량은 MAX_ANALYSIS_CHARS로 제한."""
    budget_per_source = MAX_ANALYSIS_CHARS // len(selected)
    sections = []

    if "MARKDOWN" in selected:
        sections.append(f"[페이지 본문 (마크다운)]\n{scraped.markdown[:budget_per_source]}")
    if "HTML" in selected:
        sections.append(f"[페이지 원본 HTML 소스]\n{scraped.html[:budget_per_source]}")
    if "NETWORK" in selected:
        network_text = _format_network_requests(scraped.network_requests)
        sections.append(f"[HTTP 네트워크 요청 정보]\n{network_text[:budget_per_source]}")

    return "\n\n".join(sections)


def generate_output(
    content: Union[ScrapedData, str],
    instruction: str,
    on_progress: Callable[[str], None] = log,
) -> str:
    # 옛날 호출부(문자열만 넘기는 경우)와의 호환: 마크다운만 있는 ScrapedData로 감싼다.
    scraped = content if isinstance(content, ScrapedData) else ScrapedData(markdown=str(content))

    # 1단계: 수집된 데이터 목록을 보여주고, Ollama가 필요한 것만 고르게 한다.
    on_progress("필요한 데이터 선택 중...")
    network_preview = _format_network_requests(scraped.network_requests, max_entries=15, with_headers=False)
    select_prompt = SELECT_DATA_PROMPT.format(
        markdown_chars=len(scraped.markdown),
        html_chars=len(scraped.html),
        network_count=len(scraped.network_requests),
        network_preview=network_preview or "(캡처된 네트워크 요청 없음)",
        instruction=instruction,
    )
    messages = [{"role": "user", "content": select_prompt}]
    selection_reply = _chat_turn(messages)
    if selection_reply:
        messages.append({"role": "assistant", "content": selection_reply})
    else:
        messages.pop()  # 선택 응답을 못 받으면 이 턴은 버리고 기본값(MARKDOWN)으로 진행
    selected = _parse_data_selection(selection_reply or "")
    on_progress(f"선택된 데이터: {', '.join(sorted(selected))}")

    # 2단계: 선택된 데이터만 대화에 넣고 결과물을 생성한다.
    on_progress("Ollama로 생성 중...")
    data_sections = _build_data_sections(scraped, selected)
    gen_prompt = CODE_GEN_PROMPT.format(
        data_sections=data_sections, instruction=instruction, compliance_note=COMPLIANCE_NOTE
    )
    messages.append({"role": "user", "content": gen_prompt})
    result = _chat_turn(messages)
    if not result:
        raise RuntimeError(
            "Ollama가 빈 응답을 반환했습니다. "
            "(thinking 모드 모델이 답변을 생성하지 못했을 수 있습니다. 모델/num_predict 설정을 확인하세요.)"
        )
    messages.append({"role": "assistant", "content": result})

    # 3단계: 같은 대화 맥락에서 검증·수정 반복.
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
