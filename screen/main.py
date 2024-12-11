import os
import cv2
import numpy as np
import tempfile
import streamlit as st
import json
import time

from ultralytics import YOLO
from dtaidistance import dtw
from openai import OpenAI  # OpenAI 임포트

# .venv\Scripts\activate
# streamlit run screen/main.py

icon_path = os.path.join(os.path.dirname(__file__), '../src/images/logo.jpg')

# 페이지 아이콘 설정
st.set_page_config(
    page_title="healthy_homebody",
    page_icon=icon_path,
    layout="wide"
)

# 세션 상태 초기화
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "main"

if 'selected_action' not in st.session_state:
    st.session_state.selected_action = None
    
if 'uploaded_video_path' not in st.session_state:
    st.session_state.uploaded_video_path = None

if 'description_video_path' not in st.session_state:
    st.session_state.description_video_path = None

# YOLO 모델 로드
@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8m-pose.pt', verbose=False)

# 메인 페이지
def main_page():
    st.markdown('<div class="title-style">Healthy Homebody</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-style section">필라테스 동작을 선택하세요</div>', unsafe_allow_html=True)

    actions = [
        "로우 런지(Low Lunge)",
        "파르브리타 자누 시르사아사나(Revolved Head-to-Knee Pose)",
        "선 활 자세(Standing Split)",
        "런지 사이트 스트레칭(Lunging Side Stretch)"
    ]

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
        </div>
        """, 
        unsafe_allow_html=True
    )

# 페이지 1: 동작 설명 페이지
def page1():
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
    video_path = action_info[selected_action]
    st.session_state.description_video_path = video_path  # 설명 비디오 경로 저장
    st.markdown(f'<h2 class="sub-title-style">{selected_action}</h2>', unsafe_allow_html=True)

    video_path = action_info[selected_action]["video_path"]
    if os.path.exists(video_path):
        with open(video_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes, format="video/mp4", start_time=0)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("목록으로"):
            st.session_state.selected_page = "main"

    with col2:
        if st.button("다음"):
            st.session_state.selected_page = "page2"

    description = action_info[selected_action]["description"]
    for section_title, section_content in description:
        st.markdown(f'<h3>{section_title}</h3>', unsafe_allow_html=True)
        for line in section_content:
            st.markdown(f'<li class="animated-section">{line}</li>', unsafe_allow_html=True)

# 페이지 2: 사용자 비디오 업로드 및 비교 페이지
def page2():
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
                model="gpt-4o",  # OpenAI API 모델명
                messages=messages,
                temperature=0.7
            )
            advice = result.choices[0].message.content  # GPT-4의 응답 추출
            return advice
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
            return "피드백을 생성하는 동안 문제가 발생했습니다. 다시 시도해주세요."
        
    # YOLO 모델 불러오기
    model = YOLO('yolov8m-pose.pt')  # YOLOv8 포즈 모델 경로

    # keypoints 좌표를 [0, 1]로 정규화하는 함수
    def normalize_keypoints(keypoints, frame_width, frame_height):
        normalized_keypoints = np.copy(keypoints)
        for i in range(0, len(keypoints), 2):
            normalized_keypoints[i] = keypoints[i] / frame_width  # x 좌표
            normalized_keypoints[i + 1] = keypoints[i + 1] / frame_height  # y 좌표
        return normalized_keypoints

    # Keypoints 간 상대적 거리 계산
    def calculate_relative_distances(keypoints):
        num_keypoints = len(keypoints) // 2
        relative_distances = []
        
        # 각 keypoint 사이의 거리를 계산 (유클리드 거리)
        for i in range(num_keypoints):
            for j in range(i + 1, num_keypoints):
                x1, y1 = keypoints[2 * i], keypoints[2 * i + 1]
                x2, y2 = keypoints[2 * j], keypoints[2 * j + 1]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                relative_distances.append(distance)
        
        return np.array(relative_distances)

    # Keypoints 시퀀스를 스무딩하는 함수
    def smooth_keypoints(sequence, window_size=3):
        smoothed_sequence = []
        for i in range(sequence.shape[1]):  # 각 keypoint에 대해
            smoothed = np.convolve(sequence[:, i], np.ones(window_size)/window_size, mode='valid')
            smoothed_sequence.append(smoothed)
        smoothed_sequence = np.array(smoothed_sequence).T
        return smoothed_sequence

    # 비디오에서 keypoints 추출하는 함수 (1초당 1개의 프레임만 분석)
    def extract_keypoints(video_path, model):
        cap = cv2.VideoCapture(video_path)
        keypoints_sequence = []
        max_keypoints = 34  # Keypoints 배열의 고정된 크기 (17개의 keypoints, 각 2D 좌표)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS 가져오기
        frame_interval = fps  # 1초에 한 프레임을 가져오기 위해 interval을 FPS로 설정
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:  # 1초당 1 프레임 추출
                # YOLO로 프레임에서 포즈 추출
                results = model(frame)

                for result in results:
                    if result.keypoints is not None:
                        # Keypoints 추출 (xy 좌표만 사용)
                        keypoints = result.keypoints.xy.cpu().numpy()  # NumPy 배열로 변환
                        xy_keypoints = keypoints.flatten()  # 1D로 평탄화
                        
                        # 좌표 정규화
                        normalized_keypoints = normalize_keypoints(xy_keypoints, frame_width, frame_height)

                        # Keypoints 배열의 크기를 고정 (34로 맞춤, 부족하면 0으로 패딩)
                        if len(normalized_keypoints) < max_keypoints:
                            padded_keypoints = np.zeros(max_keypoints)
                            padded_keypoints[:len(normalized_keypoints)] = normalized_keypoints
                            keypoints_sequence.append(padded_keypoints)
                        else:
                            keypoints_sequence.append(normalized_keypoints[:max_keypoints])
            
            frame_count += 1

        cap.release()

        # keypoints_sequence를 배열로 변환
        keypoints_sequence = np.array(keypoints_sequence)

        # keypoints 시퀀스에 스무딩 적용
        if len(keypoints_sequence) > 3:  # 스무딩 적용 가능한 최소 길이 확인
            keypoints_sequence = smooth_keypoints(keypoints_sequence)

        return keypoints_sequence

    # 두 시퀀스 간의 DTW 거리 계산 (상대적 거리 기반)
    def calculate_dtw_distance(seq1, seq2):
        # 각 시퀀스의 상대적 거리 계산
        seq1_relative = np.array([calculate_relative_distances(frame) for frame in seq1])
        seq2_relative = np.array([calculate_relative_distances(frame) for frame in seq2])

        # 시퀀스 길이 맞추기
        min_len = min(len(seq1_relative), len(seq2_relative))
        seq1_flat = seq1_relative[:min_len]  # 두 시퀀스의 길이를 동일하게 맞춤
        seq2_flat = seq2_relative[:min_len]

        # 프레임별로 DTW 거리 계산
        distances = []
        for i in range(min_len):
            if np.any(np.isnan(seq1_flat[i])) or np.any(np.isnan(seq2_flat[i])):
                distances.append(np.inf)  # NaN이 있는 경우, 무한대로 처리
            else:
                distance = dtw.distance(seq1_flat[i], seq2_flat[i])
                distances.append(distance)

        return np.mean(distances)

    # 두 영상의 유사도를 계산하는 메인 함수
    def compare_videos(video_path1, video_path2, model):
        st.info('첫 번째 비디오의 키포인트를 추출 중입니다...')
        keypoints_seq1 = extract_keypoints(video_path1, model)
        
        st.info('두 번째 비디오의 키포인트를 추출 중입니다...')
        keypoints_seq2 = extract_keypoints(video_path2, model)

        st.info('DTW 거리를 계산 중입니다...')
        dtw_distance = calculate_dtw_distance(keypoints_seq1, keypoints_seq2)
        
        st.success(f"두 비디오 간의 DTW 거리: {dtw_distance}")
        
            # 피드백 생성 요청
        st.info('피드백을 생성 중입니다...')
        action_name = "동작"  # 예시: 동작명을 지정하세요.
        advice = get_advice_based_on_similarity(dtw_distance, action_name)
        
        st.info('피드백:')
        st.write(advice)

        return dtw_distance



    # Streamlit 앱 UI
    st.title('비디오 포즈 유사도 비교 (DTW)')

    # 첫 번째 비디오는 고정된 경로 사용
    video_path1 = os.path.join(os.path.dirname(__file__), '../src/mp4/video6.mp4')

    # 파일 업로드 위젯 (두 번째 비디오)
    video_file_2 = st.file_uploader('두 번째 비디오 파일을 업로드하세요.', type=['mp4', 'mov', 'avi'])

    # 두 번째 파일이 업로드되었을 때 실행
    if video_file_2 is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp2:
            temp2.write(video_file_2.read())
            video_path2 = temp2.name
            
        col1, col2 = st.columns([1, 1])  
        with col1:
            if st.button('비디오 유사도 비교 시작'):
                dtw_distance = compare_videos(video_path1, video_path2, model)
                st.write(f"두 비디오 간의 유사도 (DTW 거리): {dtw_distance}")
                
        with col2:
            if st.button("다음", key="next_button", help="다음 페이지로 이동"):
                st.session_state.selected_page = "recommend_page"

        # 임시 파일 삭제
        os.remove(video_path2)
        
        
        
def recommend_page():
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
    
    if st.button("완료", key="next_button", help="메인 페이지로 이동"):
        st.session_state.selected_page = "main"

# 페이지 전환 및 실행
if st.session_state.selected_page == "main":
    main_page()
elif st.session_state.selected_page == "page1":
    page1()
elif st.session_state.selected_page == "page2":
    page2()
elif st.session_state.selected_page == "recommend_page":
    recommend_page()
    
# CSS 파일 로드 함수
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