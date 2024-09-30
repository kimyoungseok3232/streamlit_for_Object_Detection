## git clone
dataset 폴더가 있는 위치에서 아래 structure 처럼 되도록 clone

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

터미널에서 streamlit run data_analize_page.py --server.runOnSave=true 실행

이메일 입력 후 http://localhost:8501/ 로 접속하여 확인