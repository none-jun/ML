import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# 감정분석 파이프라인 로드
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="beomi/KcELECTRA-base-v2022")

sentiment_analyzer = load_sentiment_pipeline()
model = joblib.load('best_model(stacking).pkl')

model_features = [
    'major_data',
    'is_double_major',
    'major_field_IT (컴퓨터 공학 포함)',
    'major_field_IT (컴퓨터 공학 포함), 경영학',
    'major_field_IT (컴퓨터 공학 포함), 경제통상학',
    'major_field_IT (컴퓨터 공학 포함), 공학 (컴퓨터 공학 제외)',
    'major_field_IT (컴퓨터 공학 포함), 사회과학',
    'major_field_IT (컴퓨터 공학 포함), 인문학',
    'major_field_IT (컴퓨터 공학 포함), 자연과학',
    'major_field_경영학',
    'major_field_경영학, 사회과학',
    'major_field_경영학, 인문학',
    'major_field_경제통상학',
    'major_field_경제통상학, 사회과학',
    'major_field_경제통상학, 인문학',
    'major_field_공학 (컴퓨터 공학 제외)',
    'major_field_공학 (컴퓨터 공학 제외), 자연과학',
    'major_field_기타',
    'major_field_사회과학',
    'major_field_사회과학, 인문학',
    'major_field_인문학',
    'major_field_자연과학',
    'major_field_자연과학, 사회과학',
    'job_대학생',
    'job_대학원생',
    'job_직장인',
    'job_취준생',
    'sentiment_score',
    'except_job_개발/AI/R&D',
    'except_job_경영/회계',
    'except_job_공공기관',
    'except_job_금융',
    'except_job_기타',
    'except_job_기획/전략',
    'except_job_마케팅',
    'except_job_물류/무역',
    'except_job_바이오',
    'except_job_없음',
    'except_job_영업',
    'except_job_인사/HR',
    'except_job_제조/생산',
    'career_group_계획없음',
    'career_group_기타',
    'career_group_대학원',
    'career_group_취업',
    'group_네. 오프라인으로 참여하고 싶어요',
    'group_네. 온라인으로 참여하고 싶어요',
    'group_아니요. 개인적으로 학회 활동을 하고 싶어요',
    'semester_대학교 이수학기 4학기 이하',
    'semester_대학교 이수학기 5학기 이상',
    'project_개인',
    'project_팀',
    'time_0~4시간',
    'time_4~8시간',
    'time_8시간 초과',
    'incumbent_level_시니어 (10년차 ~)',
    'incumbent_level_주니어 (0~3년차)',
    'lecture_기타',
    'lecture_산업 트렌드 (예시: 챗 GPT로 인한 직무 변화)',
    'lecture_직무 강의 (예시: 실무 진행 방식 및 직무 준비생을 위한 팁)',
    'lecture_커리어 패스 과정 (예시: 비전공자/전공자의 취업 준비 및 이직 과정)',
    'whyBDA_기타',
    'whyBDA_현직자 강의',
    'whyBDA_혜택 목적',
    'whyBDA_혼자 어려움/기수 추천',
    'gain_기타',
    'gain_데이터 분석 역량',
    'gain_인적 네트워크',
    'job_경영/인사',
    'job_금융',
    'job_기타',
    'job_데이터/AI',
    'job_마케팅/기획',
    'job_미정',
    'job_엔지니어/개발',
    'interested_company_bool',
    'expected_domain_bool',
    'lecture_type_오프라인',
    'lecture_type_온,오프라인 동시',
    'lecture_type_온라인',
    'major_type_고졸',
    'major_type_기타',
    're_registration_아니요',
    're_registration_예'
]

# 한글 라벨 매핑
major_group_options = {
    "major_type_고졸": "고졸",
    "major_type_기타": "단일 전공",
    "major_type_기타2": "복수 전공"
}
jobgroup_options = {
    "job_대학생": "대학생",
    "job_대학원생": "대학원생",
    "job_직장인": "직장인",
    "job_취준생": "취준생",
    "job_경영/인사": "경영/인사",
    "job_금융": "금융",
    "job_기타": "기타",
    "job_데이터/AI": "데이터/AI",
    "job_마케팅/기획": "마케팅/기획",
    "job_미정": "미정",
    "job_엔지니어/개발": "엔지니어/개발"
}
except_group_options = {
    "except_job_개발/AI/R&D": "개발/AI/R&D",
    "except_job_경영/회계": "경영/회계",
    "except_job_공공기관": "공공기관",
    "except_job_금융": "금융",
    "except_job_기타": "기타",
    "except_job_기획/전략": "기획/전략",
    "except_job_마케팅": "마케팅",
    "except_job_물류/무역": "물류/무역",
    "except_job_바이오": "바이오",
    "except_job_없음": "없음",
    "except_job_영업": "영업",
    "except_job_인사/HR": "인사/HR",
    "except_job_제조/생산": "제조/생산"
}
career_group_options = {
    "career_group_계획없음": "계획없음",
    "career_group_기타": "기타",
    "career_group_대학원": "대학원",
    "career_group_취업": "취업"
}
groupgroup_options = {
    "group_네. 오프라인으로 참여하고 싶어요": "네. 오프라인으로 참여하고 싶어요",
    "group_네. 온라인으로 참여하고 싶어요": "네. 온라인으로 참여하고 싶어요",
    "group_아니요. 개인적으로 학회 활동을 하고 싶어요": "아니요. 개인적으로 학회 활동을 하고 싶어요"
}
semestergroup_options = {
    "semester_대학교 이수학기 4학기 이하": "대학교 이수학기 4학기 이하",
    "semester_대학교 이수학기 5학기 이상": "대학교 이수학기 5학기 이상"
}
projectgroup_options = {
    "project_개인": "개인",
    "project_팀": "팀"
}
time_group_options = {
    "time_0~4시간": "0~4시간",
    "time_4~8시간": "4~8시간",
    "time_8시간 초과": "8시간 초과"
}
incumbent_group_options = {
    "incumbent_level_시니어 (10년차 ~)": "시니어 (10년차 ~)",
    "incumbent_level_주니어 (0~3년차)": "주니어 (0~3년차)"
}
lecturegroup_options = {
    "lecture_기타": "기타",
    "lecture_산업 트렌드 (예시: 챗 GPT로 인한 직무 변화)": "산업 트렌드 (챗 GPT 등)",
    "lecture_직무 강의 (예시: 실무 진행 방식 및 직무 준비생을 위한 팁)": "직무 강의 (실무/팁)",
    "lecture_커리어 패스 과정 (예시: 비전공자/전공자의 취업 준비 및 이직 과정)": "커리어 패스 과정",
    "lecture_type_오프라인": "오프라인",
    "lecture_type_온,오프라인 동시": "온/오프라인 동시",
    "lecture_type_온라인": "온라인"
}
whybdagroup_options = {
    "whyBDA_기타": "기타",
    "whyBDA_현직자 강의": "현직자 강의",
    "whyBDA_혜택 목적": "혜택 목적",
    "whyBDA_혼자 어려움/기수 추천": "혼자 어려움/기수 추천"
}
gaingroup_options = {
    "gain_기타": "기타",
    "gain_데이터 분석 역량": "데이터 분석 역량",
    "gain_인적 네트워크": "인적 네트워크"
}
re_group_options = {
    "re_registration_아니요": "아니요",
    "re_registration_예": "예"
}

# --- 스타일 커스텀 (뒷배경 없음) ---
st.set_page_config(
    page_title="BDA 이탈 예측 서비스",
    page_icon="🎨",
    layout="centered"
)

st.markdown("""
    <style>
    .result-card {
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(44,62,80,0.08);
        margin: 24px 0;
        padding: 32px;
        font-size: 1.15em;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div>input {
        background: #f2f6fa !important;
        border-radius: 8px !important;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #6a89cc 0%, #38ada9 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
        height: 48px;
        font-size: 1.1em;
    }
    .big-title {
        font-size: 2.4em;
        font-weight: bold;
        color: #3c6382;
        margin-bottom: 0.3em;
        letter-spacing: -1px;
    }
    .desc {
        color: #576574;
        font-size: 1.1em;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">BDA 이탈 예측 서비스 🎨</div>', unsafe_allow_html=True)

selected_filters = {}

with st.form("predict_form"):
    selected_major = st.selectbox(
        "전공 유형", 
        list(major_group_options.keys()), 
        format_func=lambda x: major_group_options[x]
    )
    if selected_major == "major_type_기타2":
        selected_major = "major_type_기타"
    selected_filters["major_group"] = [selected_major]

    selected_filters["jobgroup"] = [st.selectbox(
        "현재 직무", list(jobgroup_options.keys()), format_func=lambda x: jobgroup_options[x]
    )]
    selected_filters["except_group"] = st.multiselect(
        "데이터 외 희망 직무", list(except_group_options.keys()), format_func=lambda x: except_group_options[x]
    )
    selected_filters["career_group"] = [st.selectbox(
        "희망 진로", list(career_group_options.keys()), format_func=lambda x: career_group_options[x]
    )]
    selected_filters["groupgroup"] = [st.selectbox(
        "조별활동 희망", list(groupgroup_options.keys()), format_func=lambda x: groupgroup_options[x]
    )]
    selected_filters["semestergroup"] = [st.selectbox(
        "대학교 이수학기", list(semestergroup_options.keys()), format_func=lambda x: semestergroup_options[x]
    )]
    selected_filters["projectgroup"] = [st.selectbox(
        "프로젝트 형태", list(projectgroup_options.keys()), format_func=lambda x: projectgroup_options[x]
    )]
    selected_filters["time_group"] = [st.selectbox(
        "개인공부 가능 시간", list(time_group_options.keys()), format_func=lambda x: time_group_options[x]
    )]
    selected_filters["incumbent_group"] = [st.selectbox(
        "희망 강의 현직자 연차", list(incumbent_group_options.keys()), format_func=lambda x: incumbent_group_options[x]
    )]
    selected_filters["lecturegroup"] = [st.selectbox(
        "희망 강의 유형", list(lecturegroup_options.keys()), format_func=lambda x: lecturegroup_options[x]
    )]
    selected_filters["sentiment_group"] = [st.text_input("현직자 강의 선택 이유(자유롭게 입력)")]
    selected_filters["whybdagroup"] = st.multiselect(
        "지원 동기", list(whybdagroup_options.keys()), format_func=lambda x: whybdagroup_options[x]
    )
    selected_filters["gaingroup"] = st.multiselect(
        "얻고 싶은 것", list(gaingroup_options.keys()), format_func=lambda x: gaingroup_options[x]
    )
    selected_filters["re_group"] = [st.selectbox(
        "재등록 여부", list(re_group_options.keys()), format_func=lambda x: re_group_options[x]
    )]

    submitted = st.form_submit_button("예측하기")

if submitted:
    input_data = {col: 0 for col in model_features}
    for group, selected in selected_filters.items():
        for val in selected:
            if val in input_data:
                input_data[val] = 1

    sentiment_text = selected_filters["sentiment_group"][0]
    if sentiment_text.strip():
        sentiment_result = sentiment_analyzer(sentiment_text)[0]
        if sentiment_result["label"].upper() == "POSITIVE":
            input_data["sentiment_score"] = float(sentiment_result["score"])
        elif sentiment_result["label"].upper() == "NEGATIVE":
            input_data["sentiment_score"] = -float(sentiment_result["score"])
        else:
            input_data["sentiment_score"] = 0.0
    else:
        input_data["sentiment_score"] = 0.0

    input_df = pd.DataFrame([input_data])[model_features]

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    threshold = 0.95
    if prob >= threshold:
        st.markdown(
            f"""
            <div class="result-card" style="border:2px solid #e74c3c; background:#fdecea; color:#c0392b;">
                <span style="font-size:1.5em; font-weight:bold;">🚨 이탈자입니다!</span><br>
                <span style="font-size:1.1em;">이탈 확률: <b>{prob:.2f}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card" style="border:2px solid #27ae60; background:#eafaf1; color:#145a32;">
                <span style="font-size:1.5em; font-weight:bold;">✅ 이탈자가 아닙니다!</span><br>
                <span style="font-size:1.1em;">이탈자가 아닐 확률: <b>{1-prob:.2f}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

