import json
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="데이터 분석 및 개별 이미지 확인용", layout="wide")

global categories, colors1, colors2
categories = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
colors1 = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 0, 255),  # Magenta
    (165, 42, 42),  # Brown
    (255, 192, 203) # Pink
]
colors2 = ['red', 'green', 'blue', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink']


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
    train['images']['annotation_num'] = train['annotations']['image_id'].value_counts()
    return test, train, test_data, train_data
# 출력 - train, test 데이터프레임 딕셔너리

# 데이터 페이지 단위로 데이터프레임 스플릿
# 입력 - input_df(이미지 데이터), anno_df(박스 그리기 용), rows(한번에 보여줄 데이터 수)
# 출력 - df(이미지 데이터프레임 리스트), df2(박스 그리기 용 데이터프레임 리스트)
@st.cache_data()
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

@st.cache_data()
def csv_to_dataframe(dir, csv_file):
    file_path = os.path.join(dir, csv_file)  # 파일 경로 생성
    df = pd.read_csv(file_path)  # csv 파일을 DataFrame으로 불러오기
    df['image_id'] = df['image_id'].str.extract(r"(\d+)").astype(int)
    annotation = []
    ann_id = 0
    
    # 각 파일의 내용 처리
    for row in df.itertuples(index=False, name=None):
        img = row[1]  # image_id
        pred_str = row[0]  # PredictionString

        # PredictionString이 NaN일 경우 스킵
        if pd.isna(pred_str):
            continue
        
        pred = list(map(float, pred_str.split()))
        
        # 예측값을 6개씩 묶어서 처리
        for j in range(0, len(pred), 6):
            if j + 5 >= len(pred):  # 인덱스 범위 체크
                continue
            
            category_id = int(pred[j])
            confidence = pred[j + 1]
            bbox = (pred[j + 2], pred[j + 3], pred[j + 4]-pred[j + 2], pred[j + 5]-pred[j + 3])  # (x, y, w, h)
            area = pred[j + 4] * pred[j + 5]  # 넓이 계산 (w * h)

            # annotation 리스트에 추가
            annotation.append({
                "image_id": img,
                "category_id": category_id,
                "area": area,
                "bbox": bbox,
                "isclowd": 0,
                "id": ann_id,
                "confidence": confidence
            })
            
            ann_id += 1
    
    anno = pd.DataFrame(annotation)
    
    return anno

def csv_list(output_dir):
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    return csv_files
# 팝업창 띄우기

def check_same_csv(name, csv):
    i = 1
    while name in csv:
        if i == 1:
            name = name[:-4]+'_'+str(i)+'.csv'
        else:
            name = name[:-6]+'_'+str(i)+'.csv'
        i += 1
    return name

@st.dialog("csv upload")
def upload_csv(csv):
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    # 파일이 업로드되면 처리
    if uploaded_file is not None:
        # Pandas를 사용해 CSV 파일 읽기
        df = pd.read_csv(uploaded_file)
        df = df[['PredictionString','image_id']]

        # DataFrame 내용 출력
        st.write("Data Preview:")
        st.dataframe(df)

        input_name = st.text_input("csv 파일 이름 지정", value=uploaded_file.name.replace('.csv', ''))
        if st.button("upload_csv"):
            name = check_same_csv(input_name+'.csv',csv)
            st.write("saved file name: "+name)
            df.to_csv('./output/'+name,index=False)
        if st.button("close"):
            st.rerun()

# 페이지에 있는 이미지 출력
# 입력
## type = 이미지 경로 찾을 때 사용(../dataset/train/, ../dataset/test/ 이미지 서로 다른 폴더인 경우 사용 가능),
## img_pathes = train_data['image_id'] or test_data['image_id'] 데이터프레임
## anno = train_data[['bbox','category_id']] 데이터프레임
## window = 데이터 출력할 창
def get_image(image_path, anno, transform):
    img = cv2.imread(image_path)
    tlist = [0 for _ in range(10)]
    tset = set()

    if transform:
        transformed = transform(image=img, bboxes=anno['bbox'].tolist(), labels=anno['category_id'].tolist())
        img = transformed['image']
        anno = pd.DataFrame({'bbox': transformed['bboxes'], 'category_id': transformed['labels']})

    if not anno.empty:
        for annotation,trash in anno[['bbox','category_id']].values:
            cv2.rectangle(img, np.rint(annotation).astype(np.int32), colors1[trash], 3)
            ((text_width, text_height), _) = cv2.getTextSize(categories[trash], cv2.FONT_HERSHEY_SIMPLEX, 1, 10)
            cv2.rectangle(img, (int(annotation[0]), int(annotation[1]) - int(1.3 * text_height)), (int(annotation[0] + text_width), int(annotation[1])), colors1[trash], -1)
            cv2.putText(
                img,
                text=categories[trash],
                org=(int(annotation[0]), int(annotation[1]) - int(0.3 * text_height)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, 
                color=(0,0,0), 
                lineType=cv2.LINE_AA,
            )
            tlist[trash] += 1
    for id, t in enumerate(tlist):
        if t:
            tset.add((categories[id],t))
    return img, tlist, tset

def show_images(type, img_pathes, anno, window):
    cols = window.columns(3)
    for idx,(path,id) in enumerate(img_pathes.values):
        if idx%3 == 0:
            cols = window.columns(3)
        if not anno.empty:
            img, tlist, tset = get_image(type+path, anno[anno['image_id']==id], 0)
        else:
            img, tlist, tset = get_image(type+path, pd.DataFrame(), 0)
        cols[idx%3].image(img)
        cols[idx%3].write(path)
        if tlist: 
            cols[idx%3].write(tset)
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
    pages = split_frame(img, batch_size)
    if 'annotation_num' in pages[0].columns:
        con1.dataframe(data=pages[current_page - 1][['file_name','annotation_num']], use_container_width=True)
    else:
        con1.dataframe(data=pages[current_page - 1]['file_name'], use_container_width=True)
    if not anno.empty:
        show_images(type, pages[current_page - 1][['file_name','id']], anno[['image_id','bbox','category_id']], con2)
    else:
        show_images(type, pages[current_page - 1][['file_name','id']], pd.DataFrame(), con2)

def main():
    # 원본데이터 확인 가능 아웃풋도 확인하도록 할 수 있을 듯?
    option = st.sidebar.selectbox("데이터 선택",("이미지 데이터", "원본 데이터", "트랜스폼 테스트"))

    # 데이터 로드
    testd, traind, testjson, trainjson = load_json_data()

    if option == "이미지 데이터":
        # 트레인 데이터 출력
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))

        if choose_data == "train":
            st.header("트레인 데이터")
            choose_type = st.sidebar.selectbox("시각화 선택", ("이미지 출력", "데이터 시각화"))

            if choose_type == "이미지 출력":
                text = ''
                for idx, c in enumerate(colors2):
                    text += f'<span style="color:{c};background:gray;">{categories[idx]} </span>'
                st.markdown(f'<p>{text}</p>', unsafe_allow_html=True)
                show_dataframe(traind['images'],traind['annotations'],st,'../dataset/')

            elif choose_type == "데이터 시각화":
                st.header("annotations 분석")
                st.dataframe(traind['annotations'])
    
                st.subheader("이미지당 annotation의 수")
                d = traind['annotations']['image_id'].value_counts().sort_index()
                maxd, meand, mediand, stdd = d.max(), d.mean(), d.median(), d.std()
                st.write("Annotation count max: ", maxd, "Annotation count mean: ", meand, "Annotation count median: ", mediand, "Annotation count std: ", stdd)
                st.bar_chart(d, height=400)
                st.subheader("n개의 annotation을 가진 이미지의 수")
                col1, col2 = st.columns((1,7))
                col1.write(d.value_counts().sort_index().rename('coc'))
                col2.bar_chart(d.value_counts().sort_index().rename('coc'),height=400)

                st.subheader("bbox area 분포")
                dt = traind['annotations']['area']
                d = dt.round(-3).value_counts().sort_index()
                maxd, meand, mediand, mind = dt.max(), dt.mean(), dt.median(), dt.min()
                st.write("area min: ", mind, "area max: ", maxd, "area mean: ", meand, "area median: ", mediand)
                col1, col2 = st.columns((1,7))
                col1.write(d)
                col2.bar_chart(d, height=400)

                st.subheader("cartegory 분포")
                d = traind['annotations']['category_id'].value_counts(normalize=True).sort_index()
                maxd, meand, mediand, mind = d.max(), d.mean(), d.median(), d.min()
                st.write("cartegory count min: ", mind, "cartegory count max: ", maxd, "cartegory count mean: ", meand, "cartegory count median: ", mediand)
                col1, col2 = st.columns((1,7))
                col1.write(d)
                col2.bar_chart(d, height=400)

        elif choose_data == "test":
            st.header("테스트 데이터")
            dir = 'output'
            csv = csv_list(dir)

            choose_csv = st.sidebar.selectbox("output.csv적용",("안함",)+tuple(csv))
            annotationdf = pd.DataFrame()
            if choose_csv != "안함":
                annotationdf = csv_to_dataframe(dir, choose_csv)
                testd['images']['annotation_num'] = annotationdf['image_id'].value_counts()

            show_dataframe(testd['images'],annotationdf,st,'../dataset/')

            if st.sidebar.button("csv파일 업로드"):
                upload_csv(csv)

    elif option == "원본 데이터":
        choose_data = st.sidebar.selectbox("트레인/테스트", ("train", "test"))

        if choose_data == "train":
            st.subheader("train.json")
            data = trainjson
        elif choose_data == "test":
            st.subheader("test.json")
            data = testjson

        choose_depth = st.sidebar.selectbox("출력할 key depth", ('key', 'all'))
        if choose_depth == 'all':
            st.write(data)
        elif choose_depth == 'key':
            st.write(pd.DataFrame(data.keys(), columns=[choose_data]))
            keylen = len(data.keys())
            window = st.columns(keylen)
            for idx,key in enumerate(data):
                if isinstance(data[key], dict):
                    window[idx].write(key+" : dict")
                    window[idx].write(pd.DataFrame(data[key].keys(), columns=[key]))
                elif isinstance(data[key], list) and data[key]:
                    window[idx].write(key+" : list of dict")
                    window[idx].write(pd.DataFrame(data[key][0].keys(), columns=[key]))
        # st.write(pd.DataFrame(traind.keys(),columns=[choose_data]))
        # keylen = len(traind.keys())
        # window = st.columns(keylen)
        # for idx,key in enumerate(traind):
        #     window[idx].write(traind[key].columns.rename(key))
    elif option == "트랜스폼 테스트":
        st.header("트랜스폼 테스트")
        transform_list = []
        image_data = {"1 annotation": ['../dataset/train/0000.jpg',0],
                      "big annotation": ['../dataset/train/4857.jpg',4857],
                      "8 annotation": ['../dataset/train/0008.jpg',8],
                      "34 annotation": ['../dataset/train/3049.jpg',3049],
                      "71 annotation": ['../dataset/train/4197.jpg',4197]}
        choose_data = st.sidebar.selectbox("choose_image",("1 annotation", "big annotation", "8 annotation", "34 annotation", "71 annotation"))

        if st.sidebar.checkbox("HorizontalFlip",value=True): transform_list.append(A.HorizontalFlip(p=1))
        if st.sidebar.checkbox("VerticalFlip"): transform_list.append(A.VerticalFlip(p=1))
    
        shift_x = st.sidebar.slider("Choose Horizontal Shift", min_value=-30, max_value=30, value=0, step=1)
        shift_y = st.sidebar.slider("Choose Vertical Shift", min_value=-30, max_value=30, value=0, step=1)
        s_mode = st.sidebar.radio("Choose shift Mode", options=['0', '1', '2', '3'], horizontal=True)
        transform_list.append(A.Affine(translate_percent={"x": shift_x/100, "y": shift_y/100}, scale=1, rotate=0, mode=s_mode, p=1.0))

        angle = st.sidebar.slider("Choose Rotation Angle", min_value=-180, max_value=180, value=0, step=1)
        r_mode = st.sidebar.radio("Choose Rotation Mode", options=['0', '1', '2', '3'], horizontal=True)
        transform_list.append(A.Rotate(limit=(angle, angle),border_mode=r_mode, p=1))

        transform = A.Compose(transform_list, bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        img, tlist, tset = get_image(image_data[choose_data][0],traind['annotations'][traind['annotations']['image_id']==image_data[choose_data][1]][['image_id','bbox','category_id']],transform)
        col1, col2 = st.columns((2,1))
        col1.image(img, width=550)
        if col2.checkbox("Center crop"):
            w = col2.slider('W', min_value=100, max_value=img.shape[0], value=500)
            h = col2.slider('H', min_value=100, max_value=img.shape[1], value=500)
            transform_list.append(A.CenterCrop(width=w, height=h, p=1))
            img, tlist, tset = get_image(image_data[choose_data][0],traind['annotations'][traind['annotations']['image_id']==image_data[choose_data][1]][['image_id','bbox','category_id']],transform)
            col2.image(img)
def login(password, auth):
    if password in auth:
        st.session_state['login'] = True
    else:
        st.write('need password')

if 'login' not in st.session_state or st.session_state['login'] == False:
    auth = set(['T7157','T7122','T7148','T7134','T7104','T7119'])
    password = st.sidebar.text_input('password',type='password')
    button = st.sidebar.button('login',on_click=login(password, auth))

elif st.session_state['login'] == True:
    main()