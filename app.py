"""웹사이트 데이터 추출 및 코드 생성 Streamlit 앱.

실행 방식:
    streamlit run app.py
"""

import asyncio
from datetime import datetime
from pathlib import Path

import streamlit as st

from analyzer import scrape_site, generate_output, save_result

st.set_page_config(page_title="웹사이트 분석기", page_icon="🔍", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "selected" not in st.session_state:
    st.session_state.selected = None

CODE_LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".java": "java",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".sql": "sql",
    ".sh": "bash",
    ".csv": "text",
    ".txt": "text",
}

MIME_TYPES = {
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".html": "text/html",
    ".css": "text/css",
    ".json": "application/json",
    ".java": "text/x-java-source",
    ".cs": "text/plain",
    ".go": "text/x-go",
    ".rb": "text/x-ruby",
    ".php": "application/x-httpd-php",
    ".sql": "application/sql",
    ".sh": "application/x-sh",
    ".csv": "text/csv",
    ".md": "text/markdown",
    ".txt": "text/plain",
}

st.title("🔍 웹사이트 데이터 추출 & 코드 생성기")
st.caption("URL을 스크래핑(Crawl4AI)한 뒤, 지시문에 맞는 결과물을 Ollama(qwen3.6)로 생성합니다.")

with st.sidebar:
    st.header("최근 분석 히스토리")
    if not st.session_state.history:
        st.caption("아직 분석한 내역이 없습니다.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            if st.button(
                entry["url"],
                key=f"history-{i}",
                use_container_width=True,
                help=entry["instruction"],
            ):
                st.session_state.selected = entry

url = st.text_input("분석할 URL", placeholder="https://example.com")
instruction = st.text_area(
    "지시문",
    placeholder=(
        "예: 이 사이트처럼 작동하는 견적 계산기 코드 만들어줘\n"
        "예: 상품명이랑 가격만 CSV로 뽑아줘\n"
        "예: 핵심 내용 3줄 요약해줘"
    ),
    height=100,
)
analyze_clicked = st.button("분석하기", type="primary")


def run_pipeline(url: str, instruction: str) -> tuple[str, str, Path]:
    with st.status("스크래핑 중...", expanded=True) as status:
        raw_content = asyncio.run(scrape_site(url))
        status.update(label="분석 중...")
        result = generate_output(raw_content, instruction)
        path = save_result(result)
        status.update(label="완료", state="complete")
    file_content = path.read_text(encoding="utf-8")
    return raw_content, file_content, path


if analyze_clicked:
    if not url or not url.startswith(("http://", "https://")):
        st.error("유효한 URL을 입력해주세요. (http:// 또는 https://로 시작해야 합니다)")
    elif not instruction.strip():
        st.error("지시문을 입력해주세요.")
    else:
        try:
            raw_content, file_content, path = run_pipeline(url, instruction)
        except RuntimeError as exc:
            st.error(f"오류: {exc}")
        else:
            entry = {
                "url": url,
                "instruction": instruction,
                "raw_content": raw_content,
                "result": file_content,
                "file_name": path.name,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.history.append(entry)
            st.session_state.selected = entry
            st.rerun()

selected = st.session_state.selected
if selected:
    st.subheader(selected["url"])
    st.caption(f"지시문: {selected['instruction']}")
    st.caption(f"분석 시각: {selected['timestamp']}")

    tab_result, tab_raw = st.tabs(["결과", "원본 콘텐츠"])

    with tab_result:
        ext = Path(selected["file_name"]).suffix
        if ext == ".md":
            st.markdown(selected["result"])
        else:
            st.code(selected["result"], language=CODE_LANGUAGES.get(ext, "text"))

        st.download_button(
            f"결과 다운로드 ({selected['file_name']})",
            data=selected["result"].encode("utf-8"),
            file_name=selected["file_name"],
            mime=MIME_TYPES.get(ext, "text/plain"),
        )

    with tab_raw:
        st.markdown(selected["raw_content"])
        st.download_button(
            "원본 콘텐츠 다운로드 (.md)",
            data=selected["raw_content"].encode("utf-8"),
            file_name="raw_content.md",
            mime="text/markdown",
        )
