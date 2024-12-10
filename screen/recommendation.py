import streamlit as st
import os
import json
import time
import openai
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로딩
load_dotenv()

# OpenAI API 키 로딩
openai.api_key = os.getenv("OPENAI_API_KEY")

def show():
    st.title("🧘‍♀️ 맞춤형 필라테스 계획 추천")
    st.write("""
    안녕하세요! 🧘‍♂️  
    여러분께 딱 맞는 필라테스 계획을 추천해드릴 필라테스 전문가입니다.  
    간단한 정보를 입력하시면, 나만의 필라테스 루틴을 만들어드릴게요!
    """)

    # 사용자 프로필 입력
    with st.form("user_profile"):
        st.subheader("✨ 사용자 프로필 입력")
        age = st.number_input("나이", min_value=10, max_value=100, value=25, help="정확한 추천을 위해 나이를 입력해주세요.")
        gender = st.selectbox("성별", options=["남성", "여성"], help="필라테스 동작의 강도나 추천을 맞춤화하기 위한 선택입니다.")
        weight = st.number_input("체중(kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        height = st.number_input("키(cm)", min_value=100, max_value=250, value=170)
        activity_level = st.selectbox("활동 수준", options=["낮음", "중간", "높음"], help="평소 활동량을 선택해주세요.")
        purpose = st.selectbox("필라테스를 하는 목적", options=[
            "유연성 향상", "체력 및 근력 강화", "스트레스 해소", "균형 감각 향상"
        ])

        submitted = st.form_submit_button("✨ 추천 필라테스 플랜 받기")

    if submitted:
        # 사용자 데이터 딕셔너리 생성
        user_data = {
            "age": age,
            "gender": gender,
            "weight": weight,
            "height": height,
            "activity_level": activity_level,
            "purpose": purpose
        }

        st.write("💡 입력한 프로필 데이터:")
        display_profile_insights(user_data)

        # 로딩 상태 표시
        with st.spinner('🔄 맞춤형 필라테스 플랜을 생성 중입니다...'):
            time.sleep(2)
            
            # 운동 계획 추천 생성
            recommended_plan = generate_recommendation(user_data)

        st.write(f"""
        🎉 {user_data["purpose"]}을(를) 목표로 하는 필라테스 플랜을 준비했습니다!  
        아래 추천 플랜을 확인하시고, 필라테스를 통해 목표를 성취해보세요!
        """)

        # 추천 계획 표시
        display_recommendation(recommended_plan)

def generate_recommendation(user_data):
    """
    사용자 데이터와 목적에 기반한 맞춤형 필라테스 플랜 추천 생성
    """
    # 하드코딩된 필라테스 추천 플랜 (API 호출 대신)
    recommended_plan = {
        "1주차": {
            "월요일": [
                {
                    "동작": "롤링 업 (Rolling Up)",
                    "설명": "복부 근육 강화와 척추 유연성 향상에 도움되는 기본 운동",
                    "난이도": "초급",
                    "소요시간": "10분",
                    "주의사항": "허리 통증이 있다면 조심스럽게 진행하세요"
                },
                {
                    "동작": "100s (Hundreds)",
                    "설명": "코어 근육 강화를 위한 클래식한 필라테스 동작",
                    "난이도": "중급",
                    "소요시간": "15분",
                    "주의사항": "호흡을 안정적으로 유지하세요"
                }
            ],
            "수요일": [
                {
                    "동작": "테이블 탑 (Table Top)",
                    "설명": "균형감과 코어 안정성을 향상시키는 운동",
                    "난이도": "초급",
                    "소요시간": "12분",
                    "주의사항": "손목과 무릎에 무리가 가지 않도록 주의"
                },
                {
                    "동작": "크로스 크런치 (Cross Crunch)",
                    "설명": "복부 옆면 근육을 강화하는 동작",
                    "난이도": "초급-중급",
                    "소요시간": "10분",
                    "주의사항": "목에 무리가 가지 않도록 주의"
                }
            ],
            "금요일": [
                {
                    "동작": "레그 서클 (Leg Circles)",
                    "설명": "하체 근육과 코어 안정성을 동시에 강화",
                    "난이도": "중급",
                    "소요시간": "15분",
                    "주의사항": "허리를 평평하게 바닥에 밀착시키세요"
                },
                {
                    "동작": "플랭크 변형",
                    "설명": "전신 근력 강화와 코어 안정성 향상",
                    "난이도": "중급-고급",
                    "소요시간": "10분",
                    "주의사항": "무릎이 흔들리지 않도록 주의"
                }
            ]
        }
    }
    return recommended_plan

def display_recommendation(recommendation):
    """
    추천 운동 계획을 드롭다운으로 시각적으로 표시
    """
    st.subheader("🌟 맞춤형 필라테스 주간 계획")
    
    # 추천 내용 드롭다운으로 출력
    for week, days in recommendation.items():
        st.markdown(f"### {week}")
        for day, exercises in days.items():
            with st.expander(f"**{day} 운동 계획**"):
                for exercise in exercises:
                    st.markdown(f"#### {exercise.get('동작', '운동')}")
                    st.markdown(f"**설명**: {exercise.get('설명', '없음')}")
                    st.markdown(f"**난이도**: {exercise.get('난이도', '없음')}")
                    st.markdown(f"**소요시간**: {exercise.get('소요시간', '없음')}")
                    st.markdown(f"**주의사항**: {exercise.get('주의사항', '없음')}")
                    st.divider()

def display_profile_insights(profile):
    """사용자 프로필 인사이트 시각화"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("BMI", calculate_bmi(profile['weight'], profile['height']))
        st.metric("활동 수준", profile['activity_level'])
    
    with col2:
        st.metric("목표", profile['purpose'])
        st.metric("성별", profile['gender'])

def calculate_bmi(weight, height):
    """BMI 계산"""
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return f"{bmi:.1f}"
                    
def load_css(file_path):
    """CSS 파일 내용을 읽어 반환"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"CSS 파일을 찾을 수 없습니다: {file_path}")

# CSS 파일 경로
css_path = os.path.join(os.path.dirname(__file__), '../src/styles.css')

# CSS 로드 및 적용
st.markdown(f"<style>{load_css(css_path)}</style>", unsafe_allow_html=True)