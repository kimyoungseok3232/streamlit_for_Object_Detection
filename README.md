# Streamlit for Object_Detection boostcamp level2

## Git Clone

### 개인 사용자
- `dataset` 폴더가 있는 위치에서 아래 구조처럼 프로젝트를 클론합니다:
  ```
  git clone https://github.com/kimyoungseok3232/streamlit_for_Object_Detection.git
  ```
### 팀 사용자
- 프로젝트 폴더에서 서브모듈을 추가합니다:
  ```
  git submodule add https://github.com/kimyoungseok3232/streamlit_for_Object_Detection.git
  ```
- 이후 프로젝트에 푸시하여 팀원들과 공유합니다:

### 팀원들이 서브모듈 사용 시
- 서브모듈 초기화 후 업데이트:
  ```
  git submodule init
  git submodule update
  ```

### 서브모듈 업데이트된 경우
- 최신 버전으로 업데이트:
  ```git submodule update --remote```
- 또는 서브모듈 폴더로 이동하여 수동으로 업데이트
  ```
  cd streamlit_for_Object_Detection
  git pull origin main
  ```

### 서브모듈 업데이트 후 각자 로컬에 반영
- 한 명이 서브모듈 업데이트 후 프로젝트에 푸시합니다.
- 다른 팀원은 프로젝트를 pull 받은 후 서브모듈만 업데이트합니다:
  ```
  git submodule update
  ```

## 프로젝트 구조
```
level2-objectdetection-cv-xx/
├── dataset/
│   ├── train/
│   ├── test/
│   ├── train.json
│   └── test.json
├── streamlit_for_Object_Detection/
│   ├── .git
│   ├── .gitignore
│   ├── README.md
│   ├── requirements.txt
│   └── data_analize_page.py
```

## Streamlit 실행
1. 필요한 패키지 설치:
```
pip install -r requirements.txt
```
2. Streamlit 서버 실행:
```
streamlit run data_analize_page.py --server.runOnSave=true
```
3. 브라우저에서 http://localhost:8501/ 로 접속하여 확인합니다.

![image](https://github.com/user-attachments/assets/48775dad-7bda-4020-acef-1c25aa072a86)
