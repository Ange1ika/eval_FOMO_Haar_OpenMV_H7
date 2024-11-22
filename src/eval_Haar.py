import json
import pandas as pd
from collections import defaultdict

# Порог IoU для определения пересечения объектов
IOU_THRESHOLD = 0.6

def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    xx1 = max(x1, x2)
    yy1 = max(y1, y2)
    xx2 = min(x1 + w1, x2 + w2)
    yy2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area

# Чтение данных из CSV
annotations = pd.read_csv('./data/_annotations.csv')
annotations_grouped = annotations.groupby('filename')

# Чтение данных из JSON
with open('./results/results_haar.json') as f:
    results_haar = json.load(f)

# Инициализация переменных для mAP
all_aps = []
total_tp = 0
total_fp = 0
total_fn = 0

for class_id in annotations['class'].unique():  # Перебираем все классы
    tp, fp, fn = 0, 0, 0
    all_precisions = []
    all_recalls = []
    
    for result in results_haar:
        filename = result['filename']
        detections = result['detections']


        if filename in annotations_grouped.groups:
            true_bboxes = annotations_grouped.get_group(filename)[['xmin', 'ymin', 'xmax', 'ymax']].values
            detected_bboxes = [(d['bbox']['x'], d['bbox']['y'], d['bbox']['w'], d['bbox']['h']) for d in detections if d['class_id'] == 1] # Здесь по прежнему используется class_id = 1, т.к. Хаар детектирует только лица

            matched = set()
            for true_box in true_bboxes:
                for i, det_box in enumerate(detected_bboxes):
                    if iou((true_box[0], true_box[1], true_box[2] - true_box[0], true_box[3] - true_box[1]), det_box) >= IOU_THRESHOLD: # Исправлено вычисление ширины и высоты для true_box
                        tp += 1
                        matched.add(i)
                        break

            fp += len(detected_bboxes) - len(matched)
            fn += len(true_bboxes) - len(matched)
        else:
            fp += len(detections)


        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        all_precisions.append(precision)
        all_recalls.append(recall)
    
    total_tp += tp
    total_fp += fp
    total_fn += fn

    # Сортировка по recall для вычисления AP
    sorted_indices = sorted(range(len(all_recalls)), key=lambda k: all_recalls[k], reverse=True)
    sorted_precisions = [all_precisions[i] for i in sorted_indices]
    sorted_recalls = [all_recalls[i] for i in sorted_indices]


    # Вычисление Average Precision (AP)
    ap = 0
    for i in range(1, len(sorted_recalls)):
        ap += (sorted_recalls[i-1] - sorted_recalls[i]) * sorted_precisions[i-1]
    all_aps.append(ap)


mAP = sum(all_aps) / len(all_aps) if len(all_aps) > 0 else 0
overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

print(f"mAP: {mAP:.2f}")
print(f"Precision_0.5: {overall_precision:.2f}")
print(f"Recall_0.5: {overall_recall:.2f}")