import json
import re

# Открываем файл results.txt и читаем содержимое
with open("results.txt", "r") as file:
    lines = file.readlines()

# Создаем структуру данных для хранения результатов
results = []

# Инициализация переменных
image_data = {}
detections = []
is_detecting = False

# Парсим строки файла
for line in lines:
    # Обрабатываем строку с изображением
    if line.startswith("Image:"):
        if image_data:
            image_data["detections"] = detections
            results.append(image_data)
        # Начинаем новую запись для изображения
        image_data = {
            "filename": line.split(":")[1].strip(),
            "detections": []
        }
        detections = []
        is_detecting = False

    # Обрабатываем строку с временем инференса
    elif line.startswith("Inference time:"):
        image_data["inference_time"] = line.split(":")[1].strip()

    # Обрабатываем детекции
    elif line.startswith("Detections:"):
        is_detecting = True
        continue  # Переходим к следующей строке

    # Обрабатываем конкретные детекции
    if is_detecting and line.strip():
        match = re.match(r"- Class ID: (\d+), Score: ([\d.]+), BBox: \[x=(\d+), y=(\d+), w=(\d+), h=(\d+)\]", line.strip())
        if match:
            detection = {
                "class_id": int(match.group(1)),
                "score": float(match.group(2)),
                "bbox": {
                    "x": int(match.group(3)),
                    "y": int(match.group(4)),
                    "w": int(match.group(5)),
                    "h": int(match.group(6))
                }
            }
            detections.append(detection)

# Добавляем последнюю запись
if image_data:
    image_data["detections"] = detections
    results.append(image_data)

# Сохраняем результаты в JSON файл
with open("results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Conversion completed successfully!")
