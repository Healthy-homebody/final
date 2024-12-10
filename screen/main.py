# 필라테스 동작 선택 페이지 (main)
import streamlit as st
import os

# .venv\Scripts\activate
# streamlit run screen/main.py

icon_path = os.path.join(os.path.dirname(__file__), '../src/images/logo.jpg')

# 페이지 아이콘 설정
st.set_page_config(
    page_title="healthy_homebody",
    page_icon=icon_path,
    layout="wide"
)

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


import page1
import page2
import page3
import recommendation
import dtw_test
import openai_test

# 세션 상태에서 선택된 페이지를  <sp저장할 수 있도록 초기화
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "main"

# 사이드바
page = st.sidebar.selectbox(
    "페이지를 선택하세요", 
    ["main", "page1", "page2", "page3", "recommendation", "dtw_test", "openai_test"],
    index=["main", "page1", "page2", "page3", "recommendation", "dtw_test", "openai_test"].index(st.session_state.selected_page)
)

# 메인 페이지
if page == "main":
    st.markdown('<div class="title-style">Healthy Homebody</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-title-style section">필라테스 동작을 선택하세요</div>', unsafe_allow_html=True)

    actions = [
        "로우 런지(Low Lunge)",
        "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)",
        "선 활 자세(Standing Split)",
        "런지 사이트 스트레칭(Lunging Side Stretch)"
    ]

    # 그리드 레이아웃으로 버튼 배치
    col1, col2 = st.columns(2)

    with col1:
        for action in actions[:2]:
            if st.button(action):
                st.session_state.selected_page = "page1"
                st.session_state.selected_action = action

    with col2:
        for action in actions[2:]:
            if st.button(action):
                st.session_state.selected_page = "page1"
                st.session_state.selected_action = action

    # 서비스 설명
    st.markdown(
        """
        <div class="description-style section">
        최근 비만율 증가와 함께 <span class="highlight">건강 관리</span>는 더욱 중요한 문제로 대두되고 있습니다. 
        특히 재택근무자, 집순이·집돌이, 은둔형 외톨이들은 신체 활동이 부족하여 건강이 악화될 위험이 큽니다. 
        이를 해결하기 위해 저희 서비스는 실내에서 손쉽게 할 수 있는 <span class="highlight">스트레칭 및 필라테스 동작</span>을 
        제공하여 체력 증진과 비만 예방을 목표로 하고 있습니다.
        <br><br>
        저희는 <span class="highlight">YOLOv8 포즈 추정 모델</span>을 활용하여 사용자의 운동 동작을 분석하고, 
        올바른 자세를 유지할 수 있도록<span class="highlight">정확한 피드백</span>을 제공합니다. 
        편리한 웹 기반 플랫폼을 통해 사용자는 언제 어디서나 스트레칭을 수행하고 자신의 건강 상태를 관리할 수 있습니다.
        <br><br>
        또한, 본 서비스는 재택근무자나 외출을 꺼리는 사용자들을 위한 맞춤형 솔루션으로, 
        실시간 피드백 부담 없이 <span class="highlight">자신의 영상을 업로드</span>하고 분석받을 수 있는 기능을 제공합니다.
        <br><br>
        건강을 유지하고 체력을 키우는 가장 쉽고 효과적인 방법을 저희와 함께 경험해보세요!
        </div>
        """, 
        unsafe_allow_html=True
    )



elif page == "page1":
    page1.show()
elif page == "page2":
    page2.show()
elif page == "page3":
    page3.show()
elif page == "recommendation":
    recommendation.show()
elif page == "dtw_test":
    dtw_test.compare_videos()
elif page == "openai_test":
    openai_test.get_advice_based_on_similarity()