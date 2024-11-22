import numpy as np
import pandas as pd
import json
from collections import defaultdict

# Функция для вычисления площади пересечения двух кругов (IoU)
def circle_iou(c1, c2):
    d = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    if d >= (c1[2] + c2[2]):
        return 0  # Нет пересечения
    if d <= abs(c1[2] - c2[2]):
        return 1  # Один круг полностью внутри другого
    r1, r2 = c1[2], c2[2]
    return np.arccos((r1**2 + d**2 - r2**2) / (2 * r1 * d)) + np.arccos((r2**2 + d**2 - r1**2) / (2 * r2 * d))

# Преобразование прямоугольника в круг
def bbox_to_circle(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    radius = max(xmax - xmin, ymax - ymin) / 2
    return (center_x, center_y, radius)

# Загрузка аннотаций
annotations = pd.read_csv('_annotations.csv')

# Преобразование аннотаций в круги для всех изображений с классом 'mask'
ground_truth = defaultdict(list)
for _, row in annotations.iterrows():
    if row['class'] == 'mask':
        ground_truth[row['filename']].append(bbox_to_circle(row['xmin'], row['ymin'], row['xmax'], row['ymax']))

# Загрузка результатов детекций FOMO (обычно это json файл)
with open('results.json', 'r') as f:
    fomo_detections = json.load(f)

# Преобразуем детекции в круги
detections = defaultdict(list)
for result in fomo_detections:
    for detection in result['detections']:
        if detection['class_id'] == 1 and detection['score'] > 0.5:  # Используем порог 0.5
            filename = result['filename']
            x, y, w, h = detection['bbox'].values()
            detections[filename].append({
                'circle': bbox_to_circle(x, y, x + w, y + h),
                'score': detection['score']
            })

# Функция для вычисления Precision и Recall
def calculate_precision_recall(ground_truth, detections, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0
    for filename, gt_circles in ground_truth.items():
        if filename not in detections:
            fn += len(gt_circles)  # Все аннотации - это False Negatives
            continue

        detection_list = detections[filename]
        detection_list = sorted(detection_list, key=lambda x: x['score'], reverse=True)

        detected_gt = set()
        
        for det in detection_list:
            detected_circle = det['circle']
            best_iou = 0
            best_gt = None

            # Найдем аннотацию с максимальным IoU
            for gt in gt_circles:
                iou = circle_iou(detected_circle, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            # Если IoU больше порога, считаем это верной детекцией
            if best_iou >= iou_threshold and best_gt not in detected_gt:
                tp += 1
                detected_gt.add(best_gt)
            else:
                fp += 1

        # Оставшиеся аннотации без детекций - это False Negatives
        fn += len(gt_circles) - len(detected_gt)

    # Вычисление Precision и Recall для всей выборки
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return precision, recall

# Вычисление Precision и Recall
precision, recall = calculate_precision_recall(ground_truth, detections)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")



import numpy as np
import pandas as pd
import json
from collections import defaultdict

# Функция для вычисления площади пересечения двух кругов (IoU)
def circle_iou(c1, c2):
    d = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    if d >= (c1[2] + c2[2]):
        return 0  # Нет пересечения
    if d <= abs(c1[2] - c2[2]):
        return 1  # Один круг полностью внутри другого
    r1, r2 = c1[2], c2[2]
    return np.arccos((r1**2 + d**2 - r2**2) / (2 * r1 * d)) + np.arccos((r2**2 + d**2 - r1**2) / (2 * r2 * d))

# Преобразование прямоугольника в круг
def bbox_to_circle(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    radius = max(xmax - xmin, ymax - ymin) / 2
    return (center_x, center_y, radius)

# Загрузка аннотаций
annotations = pd.read_csv('_annotations.csv')

# Преобразование аннотаций в круги для всех изображений с классом 'mask'
ground_truth = defaultdict(list)
for _, row in annotations.iterrows():
    if row['class'] == 'mask':
        ground_truth[row['filename']].append(bbox_to_circle(row['xmin'], row['ymin'], row['xmax'], row['ymax']))

# Загрузка результатов детекций FOMO (обычно это json файл)
with open('results.json', 'r') as f:
    fomo_detections = json.load(f)

# Преобразуем детекции в круги
detections = defaultdict(list)
for result in fomo_detections:
    for detection in result['detections']:
        if detection['class_id'] == 1 and detection['score'] > 0.5:  # Используем порог 0.5
            filename = result['filename']
            x, y, w, h = detection['bbox'].values()
            detections[filename].append({
                'circle': bbox_to_circle(x, y, x + w, y + h),
                'score': detection['score']
            })

# Функция для вычисления Precision, Recall и mAP
import numpy as np

# Функция для вычисления площади пересечения двух кругов (IoU)
def circle_iou(c1, c2):
    d = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    if d >= (c1[2] + c2[2]):
        return 0  # Нет пересечения
    if d <= abs(c1[2] - c2[2]):
        return 1  # Один круг полностью внутри другого
    r1, r2 = c1[2], c2[2]
    return np.arccos((r1**2 + d**2 - r2**2) / (2 * r1 * d)) + np.arccos((r2**2 + d**2 - r1**2) / (2 * r2 * d))

# Функция для вычисления Precision, Recall и mAP
def calculate_precision_recall_map(ground_truth, detections, iou_threshold=0.5):
    all_ap = []

    for filename, gt_circles in ground_truth.items():
        if filename not in detections:
            continue

        detection_list = detections[filename]
        detection_list = sorted(detection_list, key=lambda x: x['score'], reverse=True)

        tp, fp, fn = 0, 0, 0
        detected_gt = set()
        precisions, recalls = [], []

        for det in detection_list:
            detected_circle = det['circle']
            best_iou = 0
            best_gt = None

            # Найдем аннотацию с максимальным IoU
            for gt in gt_circles:
                iou = circle_iou(detected_circle, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            # Если IoU больше порога, считаем это верной детекцией
            if best_iou >= iou_threshold and best_gt not in detected_gt:
                tp += 1
                detected_gt.add(best_gt)
            else:
                fp += 1

            # Добавляем значения Precision и Recall для каждого шага
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + len(gt_circles)) if tp + len(gt_circles) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)

        # Рассчитываем Average Precision (AP) для этого изображения
        # Можно усреднить значения Precision при каждом шаге Recall
        ap = np.mean(precisions) if precisions else 0
        all_ap.append(ap)

    # Среднее значение AP по всем изображениям
    mAP = np.mean(all_ap)
    return mAP

# Вычисление mAP
mAP = calculate_precision_recall_map(ground_truth, detections)
print(f"Mean Average Precision (mAP): {mAP:.4f}")


