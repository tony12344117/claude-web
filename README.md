# claude-web

URL을 스크래핑(Crawl4AI)하고 로컬 Ollama 모델(`qwen3.6`)로 분석/가공하는 도구입니다.
CLI(`analyze.py`)와 Streamlit 웹 앱(`app.py`) 두 가지 방식으로 사용할 수 있습니다.

## 설치

```bash
pip install -r requirements.txt
```

로컬에 [Ollama](https://ollama.com/download)가 설치·실행 중이어야 하며, `qwen3.6` 모델을 미리 받아둬야 합니다.

```bash
ollama pull qwen3.6
```

## 1. CLI: 데이터 추출 + 코드 생성 도구 (`analyze.py`)

URL과 자연어 지시문을 함께 입력하면, 스크래핑한 콘텐츠를 바탕으로 지시문에 맞는
결과물(코드/CSV/요약 등 무엇이든)을 생성해 `output/` 폴더에 저장합니다.

```bash
python analyze.py <URL> "<지시문>"
```

예시:

```bash
python analyze.py https://costmyhome.com "이 사이트처럼 작동하는 견적 계산기 코드 만들어줘"
python analyze.py https://example.com "상품명이랑 가격만 CSV로 뽑아줘"
python analyze.py https://news.com "핵심 내용 3줄 요약해줘"
```

응답 형식에 따라 저장되는 파일이 자동으로 결정됩니다.

- 응답에 코드 블록(` ```python `, ` ```javascript `, ` ```html ` 등)이 있으면
  → 코드만 추출해서 `output/result.<확장자>` 로 저장 (예: `result.py`, `result.js`)
- CSV 형식(헤더 + 쉼표 구분 행)이면 → `output/result.csv`
- 그 외 일반 텍스트(요약 등)면 → `output/result.md`

## 2. Streamlit 웹 앱 (`app.py`)

URL만 입력하면 사이트 요약 / 기술스택 추정 / 콘텐츠 구조를 분석해주는 웹 UI입니다.

```bash
streamlit run app.py
```

실행하면 브라우저에서 자동으로 `http://localhost:8501` 이 열립니다.

- URL 입력 후 "분석하기" 클릭 → 스크래핑 → 분석 순으로 진행 상황 표시
- 결과를 "분석 결과" / "원본 콘텐츠" 탭으로 확인
- 각각 `.md` 파일로 다운로드 가능
- 사이드바에 세션 내 "최근 분석한 사이트" 히스토리 표시

## 파일 구성

- `analyzer.py`: 스크래핑(Crawl4AI) 및 Ollama 호출 공통 로직 (CLI/웹 앱 공용)
- `analyze.py`: CLI 진입점
- `app.py`: Streamlit 웹 앱 진입점
- `requirements.txt`: 의존 패키지 목록
