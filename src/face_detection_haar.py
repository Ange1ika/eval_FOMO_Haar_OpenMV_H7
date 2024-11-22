import sensor
import time
import image
import os

# Проверка существования директории
def dir_exists(dirpath):
    try:
        stat = os.stat(dirpath)
        # Проверяем, что это директория
        return stat[0] & 0x4000 != 0
    except OSError:
        return False

# Путь к вашей папке с валидационными изображениями
valid_images_dir = "valid"

# Reset sensor
sensor.reset()

# Sensor settings
sensor.set_contrast(3)
sensor.set_gainceiling(16)
# HQVGA и GRAYSCALE оптимальны для отслеживания лиц.
sensor.set_framesize(sensor.HQVGA)
sensor.set_pixformat(sensor.GRAYSCALE)

# Загрузка каскада Хаара
# По умолчанию используется все стадии, меньше стадий быстрее, но менее точно.
face_cascade = image.HaarCascade("frontalface", stages=25)
print(face_cascade)

# FPS clock
clock = time.clock()

# Проверяем существование директории
if not dir_exists(valid_images_dir):
    print(f"Директория {valid_images_dir} не найдена")
else:
    results = []  # Список для хранения результатов

    # Пройти по всем изображениям в каталоге
    try:
        for entry in os.ilistdir(valid_images_dir):
            image_path = valid_images_dir + "/" + entry[0]
            img = image.Image(image_path, copy_to_fb=True)
            img.to_rgb565()

            # Замер времени инференса
            start_time = time.ticks_ms()

            # Найдем объекты (лица) на изображении
            objects = img.find_features(face_cascade, threshold=0.75, scale_factor=1.25)

            # Замер времени инференса
            end_time = time.ticks_ms()
            inference_time = time.ticks_diff(end_time, start_time) / 1000.0

            # Рисуем прямоугольники вокруг обнаруженных лиц
            for r in objects:
                img.draw_rectangle(r)

            # Создаем результат для текущего изображения
            result = {
                "image_name": entry[0],
                "inference_time": inference_time,
                "detections": []
            }

            # Добавляем информацию о каждом найденном объекте
            for r in objects:
                detection = {
                    "bbox": r,
                    "class_id": 1,  # Мы предполагаем, что это всегда лицо (class_id = 1)
                    "score": 1.0  # Поскольку каскад Хаара работает без вероятности, оставим 1.0
                }
                result["detections"].append(detection)

            # Добавляем результат в список
            results.append(result)

            # Сохраняем изображение с прямоугольниками
            output_image_path = "output/" + entry[0]
            img.save(output_image_path)

    except OSError as e:
        print("Ошибка чтения директории:", e)

    # Печать результатов
    for result in results:
        print(f"Image: {result['image_name']}")
        print(f"Inference time: {result['inference_time']:.2f}s")
        print("Detections:")
        for detection in result["detections"]:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            score = detection["score"]
            print(f"  - Class ID: {class_id}, Score: {score:.2f}, BBox: {bbox}")
        print("\n")
