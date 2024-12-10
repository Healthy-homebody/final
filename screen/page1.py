import streamlit as st
import os

def show():
    st.title("동작 설명 페이지")
    
    # 동작 설명 및 동영상 정보 딕셔너리
    action_info = {
        "로우 런지(Low Lunge)": {
                "title": "로우 런지(Low Lunge)",
                "description": [
                        ("자세 설명", [
                            "제자리에서 힘을 기르는 우아한 동작이에요.",
                            "마치 춤을 추듯 부드럽게 움직이면서 몸의 균형을 잡아보세요.",
                            
                            "이 자세는 단순히 스트레칭이 아니라 내면의 힘을 키우는 여정입니다."
                            "앞쪽 무릎을 살짝 구부리면서 뒷다리는 길게 뻗어보세요.",
                            "상체는 곧게 세우고 호흡을 깊게 가져가세요.",
                            
                            "💡 초보자 팁: 처음엔 벽이나 의자를 잡고 균형을 잡으세요.",
                            "점점 안정감이 생기면 혼자 자세를 유지해보세요."
                        ]),
                        ("효과", [
                            "우리 몸의 에너지 흐름을 촉진하는 놀라운 저항 운동입니다.",
                            "고관절의 유연성을 부드럽게 확장하고 근육의 탄력성을 높여줘요.",
                            "하체 근육을 균형 있게 강화하면서 전체적인 신체 안정성을 개선해요."
                        ]),
                        ("주의사항", [
                            "자신의 신체 한계를 존중하며 천천히 진행하세요.",
                            "무리한 동작보다는 정확한 자세와 호흡에 집중해주세요.",
                            "관절에 통증이 있다면 즉시 멈추고 전문 트레이너와 상담하세요."
                        ]),
                        ("실행 방법", [
                            "1. 안정된 자세로 시작해 한 발을 앞으로 크게 내딛어요.",
                            "2. 뒷발은 살짝 들어 올려 발가락 끝으로 균형을 잡아요.",
                            "3. 앞무릎을 부드럽게 구부려 발목과 수직이 되도록 해요.",
                            "4. 상체는 곧게 세우고 깊은 호흡으로 자세의 에너지를 느껴보세요."
                        ])
                ],
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video1.mp4')
        },
        "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)": {
            "title": "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)",
            "description": [
                        ("자세 설명", [
                            "이 자세는 마치 나무가 바람에 살랑이듯 유연하게 몸을 움직이는 거예요.",
                            "한쪽 다리는 옆으로 쭉 뻗고,",
                            "다른 다리는 부드럽게 접어보세요. ",
                            
                            "상체를 뻗은 다리 쪽으로 천천히 숙이면서 반대쪽 팔은 하늘을 향해 올려보세요.",
                            "🌿 마음의 여유를 느끼며 천천히 호흡하세요. ",
                            "척추가 부드럽게 늘어나는 걸 느껴보세요."
                        ]),
                        ("효과", [
                            "척추와 근육의 깊은 이완으로 전체적인 유연성을 극대화해요.",
                            "측면 근육의 균형을 잡아 신체의 대칭성을 향상시켜줍니다.",
                            "내재된 근육 긴장을 부드럽게 풀어주는 심층적인 스트레칭 효과"
                        ]),
                        ("주의사항", [
                            "개인의 신체 조건을 충분히 고려하며 천천히 접근하세요.",
                            "허리나 관절에 통증이 있다면 전문 트레이너와 상담해주세요.",
                            "호흡과 함께 자연스러운 움직임에 집중하세요."
                        ]),
                        ("실행 방법", [
                            "1. 바닥에 편안하게 앉아 기본 자세를 잡아요.",
                            "2. 한쪽 다리는 옆으로 쭉 뻗고 다른 다리는 부드럽게 구부려요.",
                            "3. 상체를 뻗은 다리 방향으로 천천히 숙여요.",
                            "4. 반대쪽 팔은 우아하게 하늘을 향해 들어올려요."
                        ])
                    ],
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video6.mp4')
        },
        "선 활 자세(Standing Split)": {
            "title": "선 활 자세(Standing Split)",
            "description": [
                ("자세 설명", [
                    "balance의 진정한 의미를 느낄 수 있는 멋진 동작이에요.",
                    "한 발로 서서 다른 다리를 하늘 높이 들어올리는 순간, ",
                    "당신은 자신의 내면 깊은 곳의 힘을 발견하게 될 거예요.",
                    "처음엔 어렵겠지만, 연습할수록 균형과 집중력이 놀라울 정도로 향상됩니다.",
                    
                    "✨ 포인트: 시선은 앞을 고정하고, 몸의 중심을 느끼세요."
                ]),
                ("효과", [
                    "신체의 균형 감각을 섬세하게 깨우는 놀라운 자세입니다.",
                    "깊은 근육층의 안정성과 힘을 동시에 발달시켜요.",
                    "신체의 대칭성을 향상시키고 전체적인 코어 강화에 기여해요."
                ]),
                ("주의사항", [
                    "처음에는 벽이나 지지대를 활용해 안전하게 연습하세요.",
                    "개인의 유연성과 균형 수준을 존중하며 천천히 접근해요.",
                    "무릎이나 허리에 무리가 가지 않도록 주의해주세요."
                ]),
                ("실행 방법", [
                    "1. Mountain Pose에서 안정된 자세로 시작해요.",
                    "2. 한쪽 다리에 체중을 완전히 실어 균형을 잡아요.",
                    "3. 반대쪽 다리를 부드럽게, 천천히 들어 올려요.",
                    "4. 상체를 앞으로 기울이며 자세의 균형을 유지해요."
                ])
            ],
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video3.mp4')
        },
        "런지 사이트 스트레칭(Lunging Side Stretch)": {
            "title": "런지 사이트 스트레칭(Lunging Side Stretch)",
            "description": [
                ("자세 설명", [
                    "이 동작은 마치 춤추듯 우아하고 강렬해요.",
                    "한 발을 크게 내밀고 상체를 옆으로 기울이면서 온몸의 근육을 깨워보세요. ",
                    "양팔은 하늘을 향해 뻗고, ",
                    "호흡은 깊고 안정적으로 가져가세요.",

                    "🔥 근육은 이렇게 대화하듯 움직입니다. ",
                    "무릎에 무리 주지 말고, ",
                    "자신의 몸이 보내는 신호를 잘 들어보세요."
                ]),
                ("효과", [
                    "하체 근육군의 입체적이고 균형 있는 강화를 선사해요.",
                    "코어의 깊은 안정성을 향상시키는 동적 스트레칭입니다.",
                    "전체 신체의 유연성과 균형 감각을 섬세하게 개선해줘요."
                ]),
                ("주의사항", [
                    "개인의 신체 조건을 고려해 점진적으로 접근하세요.",
                    "과도한 스트레칭보다는 정확한 자세와 호흡에 집중해요.",
                    "무릎이나 관절에 통증이 있다면 즉시 중단하세요."
                ]),
                ("실행 방법", [
                    "1. 발을 어깨 너비로 안정되게 벌려 시작해요.",
                    "2. 한 발을 우아하게 앞으로 크게 내딛어 런지 자세를 취해요.",
                    "3. 양팔을 부드럽게 머리 위로 들어 올려요.",
                    "4. 호흡과 함께 상체를 측면으로 천천히 기울여요."
                ])
            ],
            "video_path": os.path.join(os.path.dirname(__file__), '../src/mp4/video4.mp4')
        }
    }
    
    selected_action = st.session_state.selected_action
    st.markdown(f'<h2 class="sub-title-style">{selected_action}</h2>', unsafe_allow_html=True)

    # Rest of the code remains similar, with some styling enhancements
    video_path = action_info[selected_action]["video_path"]
    
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0)

    # Enhanced button layout
    col1, col2 = st.columns([1, 1])  
    with col1:
        if st.button("목록", key="previous_button", help="메인 페이지로 돌아가기"):
            st.session_state.selected_page = "main"

    with col2:
        if st.button("다음", key="next_button", help="다음 페이지로 이동"):
            st.session_state.selected_page = "page2"

    # Description with enhanced styling
    description = action_info[selected_action]["description"]
    for section_title, section_content in description:
        st.markdown(f'<h3>{section_title}</h3>', unsafe_allow_html=True)
        for line in section_content:
            st.markdown(f'<li class="animated-section">{line}</li>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

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