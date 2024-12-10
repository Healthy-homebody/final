from openai import OpenAI
import streamlit as st

# OpenAI API 키 설정 (환경 변수에서 불러오기)
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# GPT-4 피드백 생성 함수
def get_advice_based_on_similarity(dtw_distance, action_name):
    user_message = (
        f"사용자와 '{action_name}' 동작을 비교한 결과, DTW 거리 값은 {dtw_distance}입니다.\n"
        "이 값에 기반하여 피드백을 제공해주세요:\n"
        "- 유사도가 낮을 경우: 자세를 교정하기 위한 구체적인 피드백 제공.\n"
        "- 유사도가 높을 경우: 칭찬과 간단한 개선점을 제안.\n"
    )
    messages = [
        {"role": "system", "content": "당신은 피트니스 전문가입니다."},
        {"role": "user", "content": user_message},
    ]
    try:
        # OpenAI API 호출
        result = client.chat.completions.create(
            model="gpt-4",  # OpenAI API 모델명
            messages=messages,
            temperature=0.7
        )
        advice = result.choices[0].message.content  # GPT-4의 응답 추출
        return advice
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        return "피드백을 생성하는 동안 문제가 발생했습니다. 다시 시도해주세요."


# Streamlit 웹 앱
st.title("💪 피트니스 동작 비교 피드백")
st.write("버튼을 누르면 OpenAI의 GPT-4로부터 피드백을 받습니다.")

# 사용자 입력값
dtw_distance = st.number_input("DTW 거리 값 입력 (예: 2)", min_value=0.0, value=2.0, step=0.1)
action_name = st.text_input("동작명 입력 (예: 스쿼트)", value="스쿼트")

# 버튼 클릭 시 GPT-4 피드백 생성
if st.button("피드백 요청"):
    with st.spinner("GPT-4에게 피드백 요청 중...⏳"):
        advice = get_advice_based_on_similarity(dtw_distance, action_name)
    
    # 결과 출력
    st.subheader("📢 GPT-4의 피드백")
    st.write(advice)