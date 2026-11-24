import csv
import html
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from google import genai
from google.genai import types
import pandas as pd
import requests
import streamlit as st

try:
    from kiwipiepy import Kiwi

    HAS_KIWI = True
except ImportError:
    Kiwi = None
    HAS_KIWI = False

try:
    import wikipediaapi

    wiki_api = wikipediaapi.Wikipedia(
        user_agent="LoanwordRefinementApp/1.0 (contact@example.com)",
        language="ko",
    )
    HAS_WIKI_LIB = True
except ImportError:
    wiki_api = None
    HAS_WIKI_LIB = False

# ---------------------------------------------------------
# 0. 환경변수(.env / api.env) 및 페이지 설정
# ---------------------------------------------------------
if Path(".env").exists():
    load_dotenv(".env")
elif Path("api.env").exists():
    load_dotenv("api.env")
else:
    load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "").strip()
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "").strip()
WOORIMALSAEM_API_KEY = os.getenv("WOORIMALSAEM_API_KEY", "").strip()

st.set_page_config(
    page_title="공공언어 외래어·신어 순화 통합 웹페이지",
    layout="wide",
)
st.title("공공언어 외래어·신어 순화 통합 웹페이지")
st.caption("NAVER API HUB · 국립국어원 우리말샘 API · Google Gemini API 기반")

REFINE_DICT_PATH = Path("refine_words.json")
WIKI_CACHE_PATH = Path("wiki_cache.json")
URIMALSAEM_FAST_CACHE_PATH = Path("urimalsaem_fast_cache.json")
URIMALSAEM_FULL_CACHE_PATH = Path("urimalsaem_full_cache.json")

PROPER_NOUN_SUFFIXES = (
    "증권", "전자", "건설", "제약", "생명", "화재", "카드", "은행", "엔터", "홀딩스", "그룹",
    "기업", "연구소", "공사", "공단", "재단", "협회", "학회", "대학", "대학교", "일보", "신문",
    "방송", "소프트", "테크", "바이오", "인베스트먼트", "자산운용", "캐피탈", "모빌리티",
    "주식회사", "(주)", "㈜",
    "축제", "페스티벌", "엑스포", "비엔날레", "한마당", "대축제", "가요제", "영화제",
    "게임", "온라인", "모바일",
)

CONTEXT_TITLES = [
    "대표", "회장", "사장", "의원", "장관", "교수", "작가", "감독", "선수", "씨",
    "주식회사", "㈜", "그룹",
]

http_session = requests.Session()
http_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
})


@st.cache_resource
def get_kiwi():
    if HAS_KIWI and Kiwi is not None:
        return Kiwi()
    return None


@st.cache_resource
def get_genai_client(api_key: str):
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


kiwi = get_kiwi()


def load_json_dict(path: Path, default: dict | None = None) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return dict(default) if default else {}


def save_json_dict(path: Path, data: dict) -> bool:
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"파일 저장 실패 ({path.name}): {e}")
        return False


def sync_refine_dict_from_disk() -> dict:
    """디스크의 refine_words.json과 세션 상태를 동기화합니다."""
    disk_data = load_json_dict(REFINE_DICT_PATH)
    if disk_data:
        st.session_state["refine_dict"].update(disk_data)
    return st.session_state["refine_dict"]


def load_any_json_file(file_path: str) -> dict:
    p = Path(file_path)
    if not p.exists():
        return {"error": f"파일을 찾을 수 없습니다: {file_path}"}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {e}"}


def search_local_refine_dict(query: str, refine_dict: dict) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(정확 일치 목록, 부분/역방향 일치 목록) 반환"""
    q = query.strip()
    if not q:
        return [], []
    q_low = q.lower()
    exact_matches = []
    partial_matches = []
    for k, v in refine_dict.items():
        k_clean = k.strip()
        v_clean = str(v).strip()
        if k_clean.lower() == q_low:
            exact_matches.append((k, v))
        elif q_low in k_clean.lower() or q_low in v_clean.lower():
            partial_matches.append((k, v))
    return exact_matches, partial_matches


# 세션 상태 초기화
if "refine_dict" not in st.session_state:
    default_words = {
        "프레젠테이션": "시청각 설명회",
        "마이그레이션": "체계 이전",
        "파이프라인": "연계 체계",
        "클라우드": "가상 저장 공간",
        "레거시": "기존 시스템",
    }
    loaded = load_json_dict(REFINE_DICT_PATH)
    if loaded:
        st.session_state["refine_dict"] = loaded
    else:
        st.session_state["refine_dict"] = default_words
        save_json_dict(REFINE_DICT_PATH, default_words)

if "wiki_cache" not in st.session_state:
    st.session_state["wiki_cache"] = load_json_dict(WIKI_CACHE_PATH)

if "urimalsaem_fast_cache" not in st.session_state:
    st.session_state["urimalsaem_fast_cache"] = load_json_dict(URIMALSAEM_FAST_CACHE_PATH)

if "urimalsaem_full_cache" not in st.session_state:
    st.session_state["urimalsaem_full_cache"] = load_json_dict(URIMALSAEM_FULL_CACHE_PATH)

if "news_api_items" not in st.session_state:
    st.session_state["news_api_items"] = []

if "candidate_info_map" not in st.session_state:
    st.session_state["candidate_info_map"] = {}

if "tab1_result" not in st.session_state:
    st.session_state["tab1_result"] = None

if "tab4_last_query" not in st.session_state:
    st.session_state["tab4_last_query"] = ""

if "custom_ref_input" not in st.session_state:
    st.session_state["custom_ref_input"] = ""

if "cand_ref_input" not in st.session_state:
    st.session_state["cand_ref_input"] = ""

if "pending_cand_ref" not in st.session_state:
    st.session_state["pending_cand_ref"] = None

if "pending_custom_ref" not in st.session_state:
    st.session_state["pending_custom_ref"] = None

if "prev_selected_cand" not in st.session_state:
    st.session_state["prev_selected_cand"] = None

if "tab3_notice" not in st.session_state:
    st.session_state["tab3_notice"] = ""

if "tab4_notice" not in st.session_state:
    st.session_state["tab4_notice"] = ""

# 위젯 인스턴스화 전 대기 값 반영
if st.session_state["pending_cand_ref"] is not None:
    st.session_state["cand_ref_input"] = st.session_state["pending_cand_ref"]
    st.session_state["pending_cand_ref"] = None

if st.session_state["pending_custom_ref"] is not None:
    st.session_state["custom_ref_input"] = st.session_state["pending_custom_ref"]
    st.session_state["pending_custom_ref"] = None


def on_candidate_change():
    """후보 단어가 바뀌면 이전 AI 추천 입력창을 초기화합니다."""
    st.session_state["cand_ref_input"] = ""


def persist_caches():
    save_json_dict(WIKI_CACHE_PATH, st.session_state["wiki_cache"])
    save_json_dict(URIMALSAEM_FAST_CACHE_PATH, st.session_state["urimalsaem_fast_cache"])
    save_json_dict(URIMALSAEM_FULL_CACHE_PATH, st.session_state["urimalsaem_full_cache"])


def clean_html_text(raw_text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", "", raw_text or "")
    return html.unescape(no_tags).strip()


# ---------------------------------------------------------
# 1. Gemini API 연동 (순화 지식 주입 + 캐싱)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_refinement_system_instruction_cached(dict_items_tuple: tuple[tuple[str, str], ...]) -> str:
    examples = dict_items_tuple[:40]
    example_lines = "\n".join([f"- {orig} -> {ref}" for orig, ref in examples])
    return (
        "당신은 국립국어원 기준을 따르는 한국어 공공언어 및 외래어·신어 순화 전문가입니다.\n"
        "핵심 원칙:\n"
        "1. 외래어, 신어, 어려운 전문용어를 일반 국민이 이해하기 쉬운 자연스러운 우리말로 바꿉니다.\n"
        "2. 아래 등록된 순화 사전의 어조와 방식을 기준으로 삼아 문맥에 알맞게 다듬습니다.\n"
        "3. 어색한 직역이나 과장된 신조어는 피하고 실제 쓰임에 자연스러운 표현을 우선합니다.\n"
        "4. 사용자가 요청한 출력 형식(순화어 1개만 출력 또는 다듬어진 문장만 출력)을 엄격히 지킵니다.\n\n"
        f"[국립국어원 및 자체 순화 사전 참고 예시]\n{example_lines}"
    )


def ask_gemini_refined(
    prompt: str,
    use_refinement_knowledge: bool = True,
    temperature: float = 0.2,
) -> tuple[str, str | None]:
    if not GEMINI_API_KEY:
        return "", "GEMINI_API_KEY가 설정되지 않았습니다. .env 또는 api.env 파일을 확인해 주세요."

    client = get_genai_client(GEMINI_API_KEY)
    if client is None:
        return "", "Gemini 클라이언트를 초기화하지 못했습니다."

    sys_inst = None
    if use_refinement_knowledge:
        dict_tuple = tuple(sorted(st.session_state["refine_dict"].items()))
        sys_inst = build_refinement_system_instruction_cached(dict_tuple)

    config = (
        types.GenerateContentConfig(system_instruction=sys_inst, temperature=temperature)
        if sys_inst
        else types.GenerateContentConfig(temperature=temperature)
    )

    models_to_try = ["gemini-3.8-flash", "gemini-2.5-flash"]
    last_err = ""
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (response.text or "").strip()
            if text:
                return text, None
            last_err = f"{model_name}: 응답이 비어 있습니다."
        except Exception as e:
            last_err = f"{model_name} ({type(e).__name__}): {e}"

    return "", last_err


# ---------------------------------------------------------
# 2. 위키백과 · 우리말샘 API 연동
# ---------------------------------------------------------
def check_wikipedia_proper_noun(word: str, wiki_cache: dict) -> bool:
    if len(word) < 2:
        return False
    if word in wiki_cache:
        return bool(wiki_cache[word])

    proper_keywords = [
        "태어남", "인물", "기업", "회사", "지명", "도시",
        "축제", "게임", "증권", "정치인", "가수", "배우",
    ]
    if HAS_WIKI_LIB and wiki_api is not None:
        try:
            page = wiki_api.page(word)
            if page.exists():
                cat_str = " ".join(list(page.categories.keys()))
                if any(kw in cat_str for kw in proper_keywords):
                    wiki_cache[word] = True
                    return True
        except Exception:
            pass

    url = (
        "https://ko.wikipedia.org/w/api.php"
        f"?action=query&titles={urllib.parse.quote(word)}&prop=categories&format=json"
    )
    try:
        res = http_session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2.5)
        if res.status_code == 200:
            data = res.json()
            pages = data.get("query", {}).get("pages", {})
            for page_id, page_info in pages.items():
                if page_id != "-1":
                    categories = page_info.get("categories", [])
                    cat_str = " ".join([c.get("title", "") for c in categories])
                    if any(kw in cat_str for kw in proper_keywords):
                        wiki_cache[word] = True
                        return True
    except Exception:
        pass

    wiki_cache[word] = False
    return False


def is_proper_noun_fast(word: str, tag: str | None = None, sentence: str = "") -> bool:
    if tag == "NNP":
        return True
    for suffix in PROPER_NOUN_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix):
            return True
    if sentence:
        for title in CONTEXT_TITLES:
            if f"{word} {title}" in sentence or f"{title} {word}" in sentence:
                return True
    return False


def check_urimalsaem_fast(word: str, fast_cache: dict) -> bool:
    if not WOORIMALSAEM_API_KEY:
        return False
    if word in fast_cache:
        return bool(fast_cache[word])

    base_url = "https://opendict.korean.go.kr/api/search"
    params = {
        "key": WOORIMALSAEM_API_KEY,
        "q": word,
        "req_type": "xml",
        "part": "word",
        "sort": "dict",
        "advanced": "y",
        "method": "exact",
    }
    try:
        res = http_session.get(f"{base_url}?{urllib.parse.urlencode(params)}", timeout=3.0)
        if res.status_code == 200:
            total = int(ET.fromstring(res.text).findtext(".//total", default="0"))
            is_reg = total > 0
            fast_cache[word] = is_reg
            return is_reg
    except Exception:
        pass
    return False


def search_urimalsaem_full(word: str, bypass_negative_cache: bool = True) -> dict:
    if not WOORIMALSAEM_API_KEY:
        return {"is_registered": False, "reason": "API_KEY_MISSING"}
    full_cache = st.session_state["urimalsaem_full_cache"]
    if word in full_cache:
        cached_val = full_cache[word]
        if cached_val.get("is_registered") or not bypass_negative_cache:
            return cached_val

    base_url = "https://opendict.korean.go.kr/api/search"
    params = {
        "key": WOORIMALSAEM_API_KEY,
        "q": word,
        "req_type": "xml",
        "part": "word",
        "sort": "dict",
        "advanced": "y",
        "method": "exact",
    }
    try:
        res = http_session.get(f"{base_url}?{urllib.parse.urlencode(params)}", timeout=4.0)
        if res.status_code == 200:
            root = ET.fromstring(res.text)
            total = int(root.findtext(".//total", default="0"))
            if total > 0:
                definitions = []
                for item in root.findall(".//item"):
                    w_el = item.findtext("word", default="").replace("-", "").replace("^", "")
                    pos = item.findtext(".//pos", default="명사")
                    defn = item.findtext(".//definition", default="").strip()
                    if w_el == word and defn:
                        definitions.append(f"[{pos}] {defn}")
                if definitions:
                    res_dict = {"is_registered": True, "word": word, "definitions": definitions}
                    full_cache[word] = res_dict
                    save_json_dict(URIMALSAEM_FULL_CACHE_PATH, full_cache)
                    return res_dict
    except Exception:
        pass

    res_dict = {"is_registered": False, "reason": "NOT_FOUND"}
    full_cache[word] = res_dict
    save_json_dict(URIMALSAEM_FULL_CACHE_PATH, full_cache)
    return res_dict


# ---------------------------------------------------------
# 3. NAVER API HUB 뉴스 검색 및 신어 후보 탐지 (빈도수 포함)
# ---------------------------------------------------------
def fetch_naver_news_api(query: str, display: int = 30, start: int = 1, sort: str = "date") -> tuple[list[dict], str | None]:
    if not NAVER_CLIENT_ID or not NAVER_CLIENT_SECRET:
        return [], "네이버 API 키(NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)가 설정되지 않았습니다."

    effective_query = query.strip() if query.strip() else "외래어 신어 순화"
    url = "https://naverapihub.apigw.ntruss.com/search/v1/news"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": effective_query,
        "display": min(max(display, 1), 100),
        "start": min(max(start, 1), 1000),
        "sort": sort,
    }
    try:
        resp = http_session.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        raw_items = resp.json().get("items", [])
        cleaned = []
        seen_links = set()
        for item in raw_items:
            link = item.get("originallink") or item.get("link") or ""
            if link and link in seen_links:
                continue
            seen_links.add(link)
            cleaned.append({
                "title": clean_html_text(item.get("title", "")),
                "description": clean_html_text(item.get("description", "")),
                "link": item.get("link", ""),
                "originallink": item.get("originallink", ""),
                "pubDate": item.get("pubDate", ""),
            })
        return cleaned, None
    except Exception as e:
        detail = ""
        if hasattr(e, "response") and e.response is not None:
            detail = f" | 응답: {e.response.text}"
        return [], f"NAVER API HUB 호출 오류: {e}{detail}"


def extract_candidate_words_from_items(
    news_items: list[dict],
    only_unregistered: bool = True,
    filter_proper: bool = True,
) -> dict[str, dict]:
    """반환 구조: {단어: {"count": 출현횟수, "contexts": [예문...]}}"""
    refine_dict = dict(st.session_state["refine_dict"])
    refine_lower_set = {k.strip().lower() for k in refine_dict.keys()}
    wiki_cache = dict(st.session_state["wiki_cache"])
    fast_cache = dict(st.session_state["urimalsaem_fast_cache"])

    sentences = []
    for item in news_items:
        if item.get("title"):
            sentences.append(item["title"])
        if item.get("description"):
            sentences.append(item["description"])

    raw_info: dict[str, dict] = {}

    for sent in sentences:
        if kiwi is not None:
            tokens = [(t.form, t.tag) for t in kiwi.tokenize(sent)]
        else:
            tokens = [(tok, None) for tok in re.findall(r"[가-힣A-Za-z]{2,}", sent)]

        for form, tag in tokens:
            word = form.strip()
            if len(word) < 2:
                continue
            if filter_proper and is_proper_noun_fast(word, tag=tag, sentence=sent):
                continue
            is_target_pos = (tag in ("NNG", "SL")) if tag is not None else True
            if (is_target_pos or re.search(r"[a-zA-Z]", word)) and word.lower() not in refine_lower_set:
                entry = raw_info.setdefault(word, {"count": 0, "contexts": []})
                entry["count"] += 1
                if sent not in entry["contexts"] and len(entry["contexts"]) < 3:
                    entry["contexts"].append(sent)

    if filter_proper and raw_info:
        cand_list = list(raw_info.keys())
        with ThreadPoolExecutor(max_workers=10) as executor:
            for w, is_proper in executor.map(
                lambda w: (w, check_wikipedia_proper_noun(w, wiki_cache)),
                cand_list,
            ):
                if is_proper:
                    raw_info.pop(w, None)

    filtered_info: dict[str, dict] = {}
    if only_unregistered and raw_info:
        rem_list = list(raw_info.keys())
        with ThreadPoolExecutor(max_workers=10) as executor:
            for w, is_neologism in executor.map(
                lambda w: (
                    w,
                    True if re.search(r"[a-zA-Z]", w) else not check_urimalsaem_fast(w, fast_cache),
                ),
                rem_list,
            ):
                if is_neologism:
                    filtered_info[w] = raw_info[w]
    else:
        filtered_info = raw_info

    st.session_state["wiki_cache"] = wiki_cache
    st.session_state["urimalsaem_fast_cache"] = fast_cache
    persist_caches()
    return filtered_info


# ---------------------------------------------------------
# 4. 치환 로직 (영단어 경계 보호 + 통계 계산)
# ---------------------------------------------------------
def replace_words(text: str) -> tuple[str, list[tuple[str, str]]]:
    refine_dict = st.session_state["refine_dict"]
    replaced_terms: list[tuple[str, str]] = []
    result_text = text

    for orig_word in sorted(refine_dict.keys(), key=len, reverse=True):
        refined_word = str(refine_dict[orig_word]).strip()
        if not orig_word or not refined_word or orig_word == refined_word:
            continue

        # 순수 영문/숫자 약어인 경우 다른 영단어 중간(예: AI in DETAIL) 오치환 방지
        if re.fullmatch(r"[A-Za-z0-9_-]+", orig_word):
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(orig_word)}(?![A-Za-z0-9])", re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(orig_word), re.IGNORECASE)

        if pattern.search(result_text):
            result_text = pattern.sub(refined_word, result_text)
            if (orig_word, refined_word) not in replaced_terms:
                replaced_terms.append((orig_word, refined_word))
    return result_text, replaced_terms


# ---------------------------------------------------------
# 5. 사이드바: API 상태, 캐시 관리 및 JSON 확인
# ---------------------------------------------------------
with st.sidebar:
    st.header("시스템 현황 및 관리")
    st.metric("등록 사전 단어 수", f"{len(st.session_state['refine_dict'])}개")

    st.markdown("**API 키 연결 상태**")
    st.write(f"- Gemini API: {'연결됨' if GEMINI_API_KEY else '미설정'}")
    st.write(f"- NAVER API HUB (뉴스): {'연결됨' if (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET) else '미설정'}")
    st.write(f"- 우리말샘 API: {'연결됨' if WOORIMALSAEM_API_KEY else '미설정'}")
    st.write(f"- Kiwi 형태소 분석기: {'사용 가능' if kiwi is not None else '폴백 모드'}")

    if st.button("API 검색 캐시 초기화", help="우리말샘/위키백과 캐시를 비우고 최신 상태로 다시 조회합니다."):
        st.session_state["wiki_cache"] = {}
        st.session_state["urimalsaem_fast_cache"] = {}
        st.session_state["urimalsaem_full_cache"] = {}
        persist_caches()
        st.success("캐시가 초기화되었습니다.")

    st.divider()
    json_path = st.text_input("JSON 파일 경로", value="refine_words.json")
    loaded_json = load_any_json_file(json_path)
    with st.expander("JSON 내용 미리보기", expanded=False):
        st.json(loaded_json)


# ---------------------------------------------------------
# 6. 메인 탭 구성
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. 외래어 순화하기",
    "2. DB 관리 & 내보내기",
    "3. 뉴스 API 신어 탐지 & AI 순화",
    "4. 사전 통합 검색 & 커스텀 순화",
])

# =========================================================
# TAB 1: 외래어 순화하기 (빠른 사전 치환 + 선택적 AI 다듬기)
# =========================================================
with tab1:
    st.subheader("문장·공문서 외래어 순화")
    sample_sentence = "오늘 발표회 프레젠테이션 준비를 위해 파이프라인 데이터의 클라우드 마이그레이션 작업을 수행했습니다."
    input_text = st.text_area("문장 입력 (여러 줄 공문서도 가능)", value=sample_sentence, height=120)

    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        use_ai_polish = st.checkbox(
            "Gemini AI로 미등록 외래어 추가 순화 및 조사·띄어쓰기 다듬기",
            value=False,
            help="체크를 끄면 등록된 사전 DB만으로 0.01초 만에 즉시 치환합니다.",
        )
    with opt_col2:
        skip_ai_if_clean = st.checkbox(
            "치환할 외래어가 없으면 AI 호출 건너뛰기 (속도 최적화)",
            value=True,
        )

    if st.button("우리말로 다듬기", type="primary", key="btn_tab1"):
        raw_in = input_text.strip()
        if not raw_in:
            st.warning("문장을 입력해 주세요.")
        else:
            sync_refine_dict_from_disk()
            stage1_text, replaced_list = replace_words(raw_in)
            final_text = stage1_text
            ai_err = None
            ai_executed = False

            has_latin = bool(re.search(r"[A-Za-z]{2,}", stage1_text))
            should_run_ai = use_ai_polish and (
                not skip_ai_if_clean or len(replaced_list) > 0 or has_latin
            )

            if should_run_ai:
                prompt = (
                    "다음 문장은 1차로 순화 사전을 적용한 문장입니다. "
                    "남아 있는 불필요한 외래어나 어려운 한자어가 있다면 쉬운 우리말로 함께 순화하고, "
                    "치환된 단어에 맞게 조사와 띄어쓰기를 자연스럽게 다듬어 최종 문장만 출력하세요.\n\n"
                    f"원문: {raw_in}\n"
                    f"1차 치환 문장: {stage1_text}"
                )
                with st.spinner("Gemini가 순화 지식 기반으로 문장을 다듬는 중입니다..."):
                    ai_res, ai_err = ask_gemini_refined(prompt, use_refinement_knowledge=True)
                    if not ai_err and ai_res:
                        final_text = ai_res
                        ai_executed = True

            st.session_state["tab1_result"] = {
                "orig": raw_in,
                "stage1": stage1_text,
                "final": final_text,
                "replaced": replaced_list,
                "error": ai_err,
                "ai_executed": ai_executed,
            }

    res1 = st.session_state.get("tab1_result")
    if res1:
        if res1.get("error"):
            st.error(f"Gemini 다듬기 오류 (1차 사전 치환 결과만 표시됩니다): {res1['error']}")

        m1, m2, m3 = st.columns(3)
        m1.metric("사전 치환 단어 수", f"{len(res1['replaced'])}개")
        m2.metric("처리 방식", "사전 + Gemini AI" if res1.get("ai_executed") else "로컬 사전 즉시 치환")
        orig_len = len(res1.get("orig") or res1.get("stage1") or "")
        m3.metric("글자 수 변화", f"{orig_len}자 → {len(res1.get('final', ''))}자")

        st.markdown("**순화 완료 문장**")
        st.success(res1["final"])

        if res1["stage1"] != res1["final"]:
            with st.expander("1단계 사전 치환 결과 비교"):
                st.write(res1["stage1"])

        st.markdown(f"**사전 기반 외래어 → 순화어 치환 내역 (총 {len(res1['replaced'])}개)**")
        if res1["replaced"]:
            st.table([{"원어(외래어)": orig, "순화어": ref} for orig, ref in res1["replaced"]])
        else:
            st.info("사전에 등록된 치환 대상 외래어가 없거나 이미 순화된 문장입니다.")


# =========================================================
# TAB 2: DB 관리(추가·수정·삭제) & CSV/JSON 입출력
# =========================================================
with tab2:
    st.subheader("순화 사전 DB 관리 (추가 · 수정 · 삭제 · 내보내기)")
    sync_refine_dict_from_disk()

    col_single, col_edit, col_csv = st.columns(3)

    with col_single:
        st.markdown("#### 1. 단어 직접 추가")
        with st.form("add_single_word_form", clear_on_submit=True):
            orig_input = st.text_input("외래어(원어)", placeholder="예: 컨설팅")
            ref_input = st.text_input("순화어(다듬은 말)", placeholder="예: 자문")
            submitted = st.form_submit_button("사전에 등록")
            if submitted:
                orig_clean = orig_input.strip()
                ref_clean = ref_input.strip()
                if not orig_clean or not ref_clean:
                    st.warning("외래어와 순화어를 모두 입력해 주세요.")
                else:
                    st.session_state["refine_dict"][orig_clean] = ref_clean
                    if save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"]):
                        st.success(f"[{orig_clean} → {ref_clean}] 등록 완료")
                        st.rerun()

    with col_edit:
        st.markdown("#### 2. 등록 단어 수정 / 삭제")
        all_keys = sorted(st.session_state["refine_dict"].keys())
        if all_keys:
            edit_target = st.selectbox("대상 단어 선택", options=all_keys, key="db_edit_select")
            new_ref_val = st.text_input(
                "새 순화어",
                value=st.session_state["refine_dict"].get(edit_target, ""),
                key="db_edit_new_val",
            )
            b_upd, b_del = st.columns(2)
            with b_upd:
                if st.button("순화어 수정", key="btn_db_update"):
                    if new_ref_val.strip():
                        st.session_state["refine_dict"][edit_target] = new_ref_val.strip()
                        save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"])
                        st.success(f"[{edit_target}] 수정 완료")
                        st.rerun()
            with b_del:
                if st.button("단어 삭제", key="btn_db_delete"):
                    st.session_state["refine_dict"].pop(edit_target, None)
                    save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"])
                    st.warning(f"[{edit_target}] 삭제 완료")
                    st.rerun()
        else:
            st.info("등록된 단어가 없습니다.")

    with col_csv:
        st.markdown("#### 3. CSV 병합 & 백업 다운로드")
        uploaded_csv = st.file_uploader("CSV 업로드 (1열: 외래어, 2열: 순화어)", type=["csv"])
        if uploaded_csv is not None and st.button("CSV 병합 실행", key="btn_import_csv"):
            raw_bytes = uploaded_csv.getvalue()
            rows = []
            for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
                try:
                    decoded = raw_bytes.decode(enc)
                    rows = list(csv.reader(decoded.splitlines()))
                    break
                except Exception:
                    continue

            added_count = 0
            for row in rows:
                if len(row) >= 2:
                    w_orig = row[0].strip()
                    w_ref = row[1].strip()
                    if w_orig and w_ref and w_orig not in ("단어_용어", "외래어", "원어"):
                        st.session_state["refine_dict"][w_orig] = w_ref
                        added_count += 1

            if save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"]):
                st.success(f"CSV 병합 완료: {added_count}개 반영")
                st.rerun()

        # CSV 내보내기 생성
        csv_buf = io.StringIO()
        writer = csv.writer(csv_buf)
        writer.writerow(["외래어", "순화어"])
        for k, v in sorted(st.session_state["refine_dict"].items()):
            writer.writerow([k, v])
        st.download_button(
            label="현재 사전 CSV 다운로드",
            data=csv_buf.getvalue().encode("utf-8-sig"),
            file_name="refine_words_export.csv",
            mime="text/csv",
        )

    st.divider()
    st.markdown("#### 현재 등록된 사전 DB 목록")
    search_db_kw = st.text_input("사전 목록 필터링", placeholder="검색어 입력...")
    sorted_items = sorted(st.session_state["refine_dict"].items())
    if search_db_kw.strip():
        kw_low = search_db_kw.strip().lower()
        sorted_items = [
            (k, v) for k, v in sorted_items
            if kw_low in k.lower() or kw_low in v.lower()
        ]

    st.dataframe(
        [{"번호": i, "외래어(원어)": k, "순화어": v} for i, (k, v) in enumerate(sorted_items, start=1)],
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TAB 3: NAVER API HUB 신어 탐지 & AI 순화
# =========================================================
with tab3:
    st.subheader("NAVER API HUB 뉴스 기반 신어·외래어 후보 탐지 & AI 순화")

    if st.session_state["tab3_notice"]:
        st.success(st.session_state["tab3_notice"])
        st.session_state["tab3_notice"] = ""

    c1, c2, c3, c4 = st.columns([3, 1, 1.5, 1.5])
    with c1:
        news_query = st.text_input("뉴스 검색어", value="디지털 전환", key="news_api_query")
    with c2:
        display_count = st.number_input("검색 건수", min_value=5, max_value=100, value=25, step=5)
    with c3:
        only_unreg = st.checkbox("우리말샘 미등재·외래어만", value=True)
    with c4:
        filter_proper = st.checkbox("고유명사 자동 제외", value=True)

    if st.button("뉴스 API로 수집 및 신어 탐지", type="primary", key="btn_fetch_news_api"):
        with st.spinner("NAVER API HUB 뉴스 조회 및 형태소·사전 분석 중..."):
            items, err_msg = fetch_naver_news_api(news_query, display=int(display_count), sort="date")
            if err_msg:
                st.error(err_msg)
            else:
                st.session_state["news_api_items"] = items
                cand_info = extract_candidate_words_from_items(
                    items,
                    only_unregistered=only_unreg,
                    filter_proper=filter_proper,
                )
                st.session_state["candidate_info_map"] = cand_info
                st.session_state["pending_cand_ref"] = ""

    news_items = st.session_state.get("news_api_items", [])
    cand_info_map = st.session_state.get("candidate_info_map", {})

    # 이미 사전에 등록된 단어는 후보 목록에서 실시간 제외
    registered_lower = {k.strip().lower() for k in st.session_state["refine_dict"].keys()}
    active_cand_info = {
        w: info for w, info in cand_info_map.items()
        if w.strip().lower() not in registered_lower
    }

    left_col, right_col = st.columns([1.3, 1])

    with left_col:
        st.markdown(f"#### 수집된 뉴스 기사 ({len(news_items)}건)")
        if news_items:
            for idx, art in enumerate(news_items, start=1):
                title_md = f"**{idx}. [{art['title']}]({art['link']})**" if art.get("link") else f"**{idx}. {art['title']}**"
                st.markdown(title_md)
                if art.get("pubDate"):
                    st.caption(art["pubDate"])
                if art.get("description"):
                    st.write(art["description"])
                st.markdown("---")
        else:
            st.info("검색어를 입력하고 '뉴스 API로 수집 및 신어 탐지' 버튼을 눌러 주세요.")

    with right_col:
        # 출현 빈도순 정렬
        sorted_cands = sorted(
            active_cand_info.keys(),
            key=lambda w: (-active_cand_info[w]["count"], w),
        )
        st.markdown(f"#### 탐지된 신어·외래어 후보 ({len(sorted_cands)}개)")

        if sorted_cands:
            selected_cand = st.selectbox(
                "후보 단어 선택 (출현 빈도순)",
                options=sorted_cands,
                format_func=lambda w: f"{w} ({active_cand_info[w]['count']}회)",
                key="selected_cand_box",
                on_change=on_candidate_change,
            )
            contexts = active_cand_info.get(selected_cand, {}).get("contexts", [])

            st.markdown("**기사 사용 맥락 (Cross-Context)**")
            if contexts:
                for c_idx, ctx in enumerate(contexts, start=1):
                    st.write(f'{c_idx}. "{ctx}"')
            else:
                st.caption("맥락 정보가 없습니다.")

            st.divider()
            st.markdown("**Gemini AI 순화어 추천 & 사전 등록**")
            if st.button("Gemini AI 순화어 추천", key="btn_cand_ai_suggest"):
                ctx_block = "\n".join([f'- "{c}"' for c in contexts]) if contexts else "- (맥락 정보 없음)"
                prompt = (
                    f"대상 단어: {selected_cand}\n"
                    f"기사 사용 맥락:\n{ctx_block}\n\n"
                    "위 외래어 또는 신어를 대신할 가장 자연스러운 공공언어 우리말 순화어 1개만 단어로 출력하세요. "
                    "설명이나 따옴표 없이 순화어 단어만 출력하세요."
                )
                with st.spinner("Gemini 순화어 생성 중..."):
                    sug_text, sug_err = ask_gemini_refined(prompt, use_refinement_knowledge=True)
                if sug_err:
                    st.error(f"Gemini 순화어 추천 실패: {sug_err}")
                else:
                    clean_sug = sug_text.replace("순화어:", "").strip().splitlines()[0].strip("\"'")
                    st.session_state["pending_cand_ref"] = clean_sug
                    st.rerun()

            st.text_input("등록할 순화어", key="cand_ref_input")

            if st.button("사전에 등록", key="btn_save_cand"):
                ref_clean = st.session_state.get("cand_ref_input", "").strip()
                if not ref_clean:
                    st.warning("순화어를 입력하거나 AI 추천을 받아주세요.")
                else:
                    st.session_state["refine_dict"][selected_cand] = ref_clean
                    save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"])
                    st.session_state["candidate_info_map"].pop(selected_cand, None)
                    st.session_state["pending_cand_ref"] = ""
                    st.session_state["tab3_notice"] = f"[{selected_cand} → {ref_clean}] 사전에 등록되었습니다."
                    st.rerun()
        else:
            st.info("표시할 신어 후보가 없습니다.")


# =========================================================
# TAB 4: 사전 통합 검색 & 커스텀 단어 순화
# =========================================================
with tab4:
    st.subheader("사전 통합 검색 (자체 DB · 우리말샘 API · Gemini 표제항 생성)")
    sync_refine_dict_from_disk()

    if st.session_state["tab4_notice"]:
        st.success(st.session_state["tab4_notice"])
        st.session_state["tab4_notice"] = ""

    t4_col1, t4_col2 = st.columns([3, 1])
    with t4_col1:
        target_word = st.text_input("조회·순화할 단어 입력", value="프레젠테이션", key="tab4_word_input")
    with t4_col2:
        st.write("")
        st.write("")
        lookup_btn = st.button("통합 사전 조회", type="primary", key="btn_tab4_lookup")

    w_clean = target_word.strip()

    # 입력한 단어에 대해 자체 DB 등록 정보는 버튼 클릭 전에도 항상 실시간 표시
    if w_clean:
        exact_hits, partial_hits = search_local_refine_dict(w_clean, st.session_state["refine_dict"])
        if exact_hits:
            st.markdown("### [자체 순화 사전 등록어 — 정확 일치]")
            for orig_k, ref_v in exact_hits:
                st.success(f"**{orig_k}** → **{ref_v}**")
        elif partial_hits:
            st.markdown("### [자체 순화 사전 등록어 — 유사·부분 일치]")
            st.table([{"원어(외래어)": k, "순화어": v} for k, v in partial_hits])
        else:
            st.info(f"자체 순화 사전 DB에 '{w_clean}' 항목은 아직 등록되어 있지 않습니다.")

    if lookup_btn:
        if not w_clean:
            st.warning("단어를 입력해 주세요.")
        else:
            with st.spinner("국립국어원 우리말샘 API 조회 중..."):
                uri_res = search_urimalsaem_full(w_clean, bypass_negative_cache=True)

            if uri_res.get("is_registered"):
                defs_md = "\n".join([f"{i}. {d}" for i, d in enumerate(uri_res["definitions"], start=1)])
                st.session_state["tab4_external_md"] = f"### [국립국어원 우리말샘 등재 정보]\n{defs_md}"
            else:
                with st.spinner("우리말샘 미등재어 — Gemini 사전 표제항 미시구조 생성 중..."):
                    micro_prompt = (
                        f"신어 또는 외래어 '{w_clean}'의 국어사전 표제항 미시구조를 아래 포맷으로 작성해 주세요.\n"
                        f"# {w_clean}\n어종: 신어\n형성 양상: \n뜻풀이: \n[순화어 제안]: "
                    )
                    micro_text, micro_err = ask_gemini_refined(micro_prompt, use_refinement_knowledge=True)
                if micro_err:
                    st.error(f"Gemini 표제항 생성 실패: {micro_err}")
                else:
                    st.session_state["tab4_external_md"] = (
                        f"### [우리말샘 미등재 — Gemini 사전 표제항 초안]\n{micro_text}"
                    )
            st.session_state["tab4_last_query"] = w_clean

    if st.session_state.get("tab4_external_md") and st.session_state.get("tab4_last_query") == w_clean:
        st.markdown("---")
        st.markdown(st.session_state["tab4_external_md"])

    st.divider()
    st.markdown("#### 즉석 순화어 추천 및 내 사전에 등록")
    if "custom_target_input" not in st.session_state:
        st.session_state["custom_target_input"] = w_clean
    elif st.session_state.get("_last_top_word") != w_clean:
        st.session_state["custom_target_input"] = w_clean
    st.session_state["_last_top_word"] = w_clean

    c_sug1, c_sug2, c_sug3 = st.columns([2, 2, 1.5])
    with c_sug1:
        custom_target_val = st.text_input("대상 단어 (직접 수정 가능)", key="custom_target_input")
    with c_sug2:
        st.text_input("추천·입력 순화어", key="custom_ref_input")
    with c_sug3:
        st.write("")
        st.write("")
        if st.button("Gemini 즉석 추천", key="btn_custom_ai"):
            custom_w_clean = custom_target_val.strip()
            if not custom_w_clean:
                st.warning("먼저 대상 단어를 입력해 주세요.")
            else:
                uri_res = search_urimalsaem_full(custom_w_clean, bypass_negative_cache=False)
                defn_ctx = ""
                if uri_res.get("is_registered"):
                    defn_ctx = "\n[우리말샘 뜻풀이 참고]: " + " / ".join(uri_res["definitions"])
                prompt = (
                    f"외래어/전문어 '{custom_w_clean}'{defn_ctx}\n"
                    f"이 단어를 공공언어에서 사용할 수 있는 가장 자연스럽고 알맞은 우리말 순화어 1개만 단어로 출력하세요. "
                    f"부연 설명 없이 순화어 단어만 출력하세요."
                )
                with st.spinner("Gemini 순화어 생성 중..."):
                    sug_text, sug_err = ask_gemini_refined(prompt, use_refinement_knowledge=True)
                if sug_err:
                    st.error(f"Gemini 즉석 추천 실패: {sug_err}")
                else:
                    clean_sug = sug_text.strip().splitlines()[0].strip("\"'")
                    st.session_state["pending_custom_ref"] = clean_sug
                    st.rerun()

    if st.button("내 사전에 바로 등록", key="btn_custom_save"):
        custom_w_clean = st.session_state.get("custom_target_input", "").strip()
        ref_clean = st.session_state.get("custom_ref_input", "").strip()
        if not custom_w_clean or not ref_clean:
            st.warning("단어와 순화어를 모두 확인해 주세요.")
        else:
            st.session_state["refine_dict"][custom_w_clean] = ref_clean
            if save_json_dict(REFINE_DICT_PATH, st.session_state["refine_dict"]):
                st.session_state["pending_custom_ref"] = ""
                st.session_state["tab4_notice"] = f"[{custom_w_clean} → {ref_clean}] 단어가 사전 DB에 저장되었습니다."
                st.rerun()
