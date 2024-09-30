import streamlit as st
import pandas as pd
import json
import cv2
import numpy as np
st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

global categories 
categories = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

# json 파일에서 각 key 별로 데이터 불러와서 dataframe으로 변환 후 리스트에 넣고 리스트 반환
# 입력 - json 파일
# 출력 - 데이터프레임 딕셔너리
def read_data_from_json_by_columns(filename):
    data = {}
    for key in filename:
        if type(filename[key]) == list:
            data[key] = pd.DataFrame(filename[key])
        else:
            data[key] = pd.DataFrame([filename[key]])
    return data

@st.cache_data
def load_json_data():
    with open('../dataset/train.json') as t:
        train_data = json.loads(t.read())
    with open('../dataset/test.json') as t:
        test_data = json.loads(t.read())
    test = read_data_from_json_by_columns(test_data)
    train = read_data_from_json_by_columns(train_data)
    return test, train
# 출력 - train, test 데이터프레임 딕셔너리

# 데이터 페이지 단위로 데이터프레임 스플릿
# 입력 - input_df(이미지 데이터), anno_df(박스 그리기 용), rows(한번에 보여줄 데이터 수)
# 출력 - df(이미지 데이터프레임 리스트), df2(박스 그리기 용 데이터프레임 리스트)
@st.cache_data(show_spinner=False)
def split_frame(input_df, anno_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    if not anno_df.empty:
        df2 = [anno_df[(i<=anno_df['image_id'])&(anno_df['image_id']<=i+rows)] for i in range(0, len(input_df), rows)]
        return df, df2
    return df, []

# 팝업창 띄우기
@st.dialog("image")
def show_image(type,path):
    pass

# 페이지에 있는 이미지 출력
# 입력
## type = 이미지 경로 찾을 때 사용(../dataset/train/, ../dataset/test/ 이미지 서로 다른 폴더인 경우 사용 가능),
## img_pathes = train_data['image_id'] or test_data['image_id'] 데이터프레임
## anno = train_data[['bbox','category_id']] 데이터프레임
## window = 데이터 출력할 창
def show_images(type, img_pathes, anno, window):
    cols = window.columns(3)
    for idx,(path,id) in enumerate(img_pathes.values):
        if idx%3 == 0:
            cols = window.columns(3)
        img = cv2.imread(type+path)
        tlist = set()
        if not anno.empty:
            for annotation,trash in anno[anno['image_id']==id][['bbox','category_id']].values:
                cv2.rectangle(img, np.rint(annotation).astype(np.int32), (255,0,0), 2)
                tlist.add(categories[trash])
        cols[idx%3].image(img)
        cols[idx%3].write(path)
        if tlist: cols[idx%3].write(tlist)

# 데이터 프레임 페이지 단위로 출력
# 입력
## img = train_data or test_data에서 'images'
## anno = train_data or test_data에서 'annotations'
## window = 데이터 프레임 출력할 위치
## type = 이미지 경로
def show_dataframe(img,anno,window,type):
    # 가장 윗부분 데이터 정렬할 지 선택, 정렬 시 무엇으로 정렬할지, 오름차순, 내림차순 선택
    top_menu = window.columns(3)
    with top_menu[0]:
        sort = st.radio("Sort Data", options=["Yes", "No"], horizontal=1, index=1, key=[type,window,1])
    if sort == "Yes":
        with top_menu[1]:
            sort_field = st.selectbox("Sort By", options=img.columns, key=[type,window,2])
        with top_menu[2]:
            sort_direction = st.radio(
                "Direction", options=["⬆️", "⬇️"], horizontal=True
            )
        img = img.sort_values(
            by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True
        )
    # 데이터 크기 출력
    total_data = img.shape
    with top_menu[0]:
        st.write("data_shape: "+str(total_data))
    con1,con2 = window.columns((1,3))

    # 아래 부분 페이지당 데이터 수, 페이지 선택
    bottom_menu = window.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[9, 15, 27], key=[type,window,3])
    with bottom_menu[1]:
        total_pages = (
            int(len(img) / batch_size) if int(len(img) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")
    pages, anno_data = split_frame(img, anno, batch_size)
    con1.dataframe(data=pages[current_page - 1]['file_name'], use_container_width=True)
    if anno_data:
        show_images(type, pages[current_page - 1][['file_name','id']], anno_data[current_page-1][['image_id','bbox','category_id']], con2)
    else:
        show_images(type, pages[current_page - 1][['file_name','id']], pd.DataFrame(), con2)

# 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
option = st.sidebar.selectbox("데이터 선택",("원본 데이터", "결과 데이터"))

if option == "원본 데이터":
    # 데이터 로드
    testd, traind = load_json_data()
    # 트레인 데이터 출력
    choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))
    if choose_data == "train":
        st.header("트레인 데이터")
        page = show_dataframe(traind['images'],traind['annotations'],st,'../dataset/')
    elif choose_data == "test":
        st.header("테스트 데이터")
        page = show_dataframe(testd['images'],testd['annotations'],st,'../dataset/')

    