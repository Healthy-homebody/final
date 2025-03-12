# Healthy-Homebody: AI 필라테스 홈트레이너 서비스

이 프로젝트는 AI 기반 필라테스 홈트레이너 서비스를 제공합니다. 아래의 단계를 따라 환경 설정과 실행을 진행하실 수 있습니다.<br/><br/><br/>

## 1. 가상 환경 설정

### 1.1 가상 환경 생성
프로젝트에 필요한 종속성(라이브러리)을 관리하기 위해 Python 가상 환경을 생성합니다.
```
python -m venv .venv
```
위 명령어를 실행하면 `.venv` 폴더가 생성되고, 이 폴더 내에서 프로젝트에 필요한 모든 라이브러리를 설치하게 됩니다.<br/><br/><br/>

### 1.2 가상 환경 활성화
가상 환경을 활성화하려면 아래 명령어를 실행하세요:

**Windows (cmd, PowerShell):**
```
.venv\Scripts\activate
```

**macOS/Linux:**
```
source .venv/bin/activate
```
가상 환경이 활성화되면, 터미널에 `(venv)` 표시가 나타납니다.  <br/><br/><br/>

### 1.3 종속성 설치
가상 환경이 활성화된 상태에서 필요한 패키지들을 설치합니다.
```
pip install -r requirements.txt
```
위 명령어를 실행하여 `requirements.txt` 파일에 명시된 모든 라이브러리를 설치하세요.    <br/><br/><br/><br/>

## 2. 서비스 실행

### 2.1 앱 실행
모든 설정이 완료되면, 이제 Streamlit 앱을 실행할 수 있습니다.
```
streamlit run screen/main.py
```
실행 후, 터미널에 표시되는 URL (일반적으로 `http://localhost:8501` 또는 비슷한 주소)을 브라우저에 입력하면, Healthy-Homebody AI 필라테스 홈트레이너 앱을 확인할 수 있습니다.<br/><br/><br/><br/>

## 3. 프로젝트 종료
작업을 마친 후에는 가상 환경을 비활성화할 수 있습니다:
```
deactivate
```
