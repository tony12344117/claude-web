"""웹사이트 분석기 Streamlit 앱.

실행 방식:
    streamlit run app.py
"""

import asyncio
from datetime import datetime

import streamlit as st

from analyzer import scrape_site, analyze_content

st.set_page_config(page_title="웹사이트 분석기", page_icon="🔍", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []
if "selected" not in st.session_state:
    st.session_state.selected = None

st.title("🔍 웹사이트 분석기")
st.caption("URL을 스크래핑(Crawl4AI)한 뒤 Ollama(qwen3.6)로 분석합니다.")

with st.sidebar:
    st.header("최근 분석한 사이트")
    if not st.session_state.history:
        st.caption("아직 분석한 사이트가 없습니다.")
    else:
        for i, entry in enumerate(reversed(st.session_state.history)):
            if st.button(entry["url"], key=f"history-{i}", use_container_width=True):
                st.session_state.selected = entry

url = st.text_input("분석할 URL", placeholder="https://example.com")
analyze_clicked = st.button("분석하기", type="primary")


def run_analysis(url: str) -> tuple[str, str]:
    with st.status("스크래핑 중...", expanded=True) as status:
        raw_content = asyncio.run(scrape_site(url))
        status.update(label="분석 중...")
        analysis = analyze_content(raw_content)
        status.update(label="완료", state="complete")
    return raw_content, analysis


if analyze_clicked:
    if not url or not url.startswith(("http://", "https://")):
        st.error("유효한 URL을 입력해주세요. (http:// 또는 https://로 시작해야 합니다)")
    else:
        try:
            raw_content, analysis = run_analysis(url)
        except RuntimeError as exc:
            st.error(f"오류: {exc}")
        else:
            entry = {
                "url": url,
                "raw_content": raw_content,
                "analysis": analysis,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state.history.append(entry)
            st.session_state.selected = entry
            print(f"[DEBUG] appended {entry['url']}, history now={[e['url'] for e in st.session_state.history]}", flush=True)
            st.rerun()

selected = st.session_state.selected
if selected:
    st.subheader(selected["url"])
    st.caption(f"분석 시각: {selected['timestamp']}")

    tab_analysis, tab_raw = st.tabs(["분석 결과", "원본 콘텐츠"])

    with tab_analysis:
        st.markdown(selected["analysis"])
        st.download_button(
            "분석 결과 다운로드 (.md)",
            data=selected["analysis"].encode("utf-8"),
            file_name="analysis.md",
            mime="text/markdown",
        )

    with tab_raw:
        st.markdown(selected["raw_content"])
        st.download_button(
            "원본 콘텐츠 다운로드 (.md)",
            data=selected["raw_content"].encode("utf-8"),
            file_name="raw_content.md",
            mime="text/markdown",
        )
