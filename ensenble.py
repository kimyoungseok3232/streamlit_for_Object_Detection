import numpy as np
import pandas as pd
from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion
from pycocotools.coco import COCO


#submission_df 는 제출 가능한 csv를 pd로 읽은 df들의 리스트
def ensenble(submission_df, type, iou_thr, thresh):
    image_ids = submission_df[0]['image_id'].tolist()
    annotation = '../dataset/test.json'
    coco = COCO(annotation)

    prediction_strings = []
    file_names = []

    # 각 image id 별로 submission file에서 box좌표 추출
    for i, image_id in enumerate(image_ids):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]
        # 각 submission file 별로 prediction box좌표 불러오기
        for df in submission_df:

            predict_string = df[df['image_id'] == image_id]['PredictionString']
            if predict_string.empty:
                print(image_id)
                continue
            predict_string = predict_string.tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list)==0 or len(predict_list)==1:
                continue
    
            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                # box의 각 좌표를 float형으로 변환한 후 image의 넓이와 높이로 각각 정규화
                image_width = image_info['width']
                image_height = image_info['height']
                box[0] = float(box[0]) / image_width
                box[1] = float(box[1]) / image_height
                box[2] = float(box[2]) / image_width
                box[3] = float(box[3]) / image_height
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        # 예측 box가 있다면 이를 ensemble 수행
        if len(boxes_list):
            # ensemble_boxes에서 import한 nms()를 사용하여 NMS 계산 수행
            # 👉 위의 코드에서 만든 정보들을 함수에 간단하게 적용해보세요!
            # nms에 필요한 인자: [NMS할 box의 lists, confidence score의 lists, label의 list, iou에 사용할 threshold]
            if type == 0:
                boxes, scores, labels = nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr)
            elif type == 1:
                boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, iou_thr=iou_thr, thresh=thresh)
            elif type == 2:
                boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=iou_thr, skip_box_thr=thresh)
            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + ' ' + str(box[0] * image_info['width']) + ' ' + str(box[1] * image_info['height']) + ' ' + str(box[2] * image_info['width']) + ' ' + str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    return submission