## git clone
### 개인
개인으로 사용하는 경우 dataset 폴더가 있는 위치에서 아래 structure 처럼 되도록 clone

### 팀
팀으로 사용하는 경우 동일한 위치에서 최초에

git submodule add https://github.com/kimyoungseok3232/streamlit_for_Object_Detection.git

로 받아온 후 프로젝트에 push 하면 사용 가능

다른 팀원이 서브모듈 사용하는 경우 

git submodule init 후

git submodule update 하면 각자 로컬로 가져올 수 있음

submodule 업데이트 된 경우 git submodule update --remote 로 업데이트

로컬에서 수정한 경우 git submodule update --remote --merge로 충돌해결

한 명이 서브모듈 업데이트 후 프로젝트에 push 하면 다른 팀원은 프로젝트 pull 한 뒤

git submodule update 만으로 업데이트 가능

## Structure

```
level2-objectdetection-cv-xx/|
|
|─── dataset
|   |─── train
|   |─── test
|   |─── train.json
|   └─── test.json
|   
|─── streamlit_for_Object_Detection
    |─── .git
    |─── .gitignore
    |─── README.md
    |─── requirements.txt
    └─── data_analize_page.py
```
## Streamlit run

pip install -r requirements.txt

터미널에서 streamlit run data_analize_page.py --server.runOnSave=true 실행

이메일 입력 후 http://localhost:8501/ 로 접속하여 확인

![image](https://github.com/user-attachments/assets/48775dad-7bda-4020-acef-1c25aa072a86)