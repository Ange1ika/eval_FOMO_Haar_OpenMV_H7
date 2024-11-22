import time
import ml
from ml.utils import NMS
import math
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
valid_images_dir = "valid"  # Убедитесь, что добавлен завершающий слеш

# Загрузка модели
model = ml.Model("fomo_face_detection")  # Укажите вашу модель
print("Модель загружена:", model)

min_confidence = 0.4
threshold_list = [(math.ceil(min_confidence * 255), 255)]

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

# Постобработка результатов модели
def fomo_post_process(model, inputs, outputs):
    n, oh, ow, oc = model.output_shape[0]
    nms = NMS(ow, oh, inputs[0].roi)
    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            nms.add_bounding_box(x, y, x + w, y + h, score, i)
    return nms.get_bounding_boxes()

# Список для хранения предсказаний и времен инференса
results = []

# Проверяем существование директории
if not dir_exists(valid_images_dir):
    print(f"Директория {valid_images_dir} не найдена")
else:
    # Пройти по всем изображениям в каталоге
    try:
        for entry in os.ilistdir(valid_images_dir):
            image_path = valid_images_dir + "/" + entry[0]
            img = image.Image(image_path, copy_to_fb=True)
            img.to_rgb565()

            # Замер времени инференса
            start_time = time.ticks_ms()

            # Прогнозируем объекты с помощью модели
            detections = []
            for i, detection_list in enumerate(model.predict([img], callback=fomo_post_process)):
                if i == 0:  # Пропустить класс фона
                    continue
                if len(detection_list) == 0:
                    continue

                for (x, y, w, h), score in detection_list:
                    center_x = int(x + (w / 2))
                    center_y = int(y + (h / 2))
                    img.draw_circle((center_x, center_y, 12), color=colors[i])

                    # Добавляем данные предсказаний
                    detections.append({
                        "bbox": [x, y, w, h],
                        "class_id": i,
                        "score": float(score)
                    })

            # Замер времени инференса
            end_time = time.ticks_ms()
            inference_time = time.ticks_diff(end_time, start_time) / 1000.0
            #print(f"Inference time for {entry[0]}: {inference_time:.2f}s")

            # Сохранение обработанного изображения
            output_image_path = "output/" + entry[0]
            img.save(output_image_path)

            # Сохранение данных для текущего изображения
            results.append({
                "image_name": entry[0],
                "inference_time": inference_time,
                "detections": detections
            })
    except OSError as e:
        print("Ошибка чтения директории:", e)
    # # Сохранение всех данных в текстовый файл
    # output_txt_path = "inference_results.txt"

    # try:
    #     with open(output_txt_path, "w") as txt_file:
    #         for result in results:
    #             # Заголовок для каждого изображения
    #             txt_file.write("Image: {}\n".format(result["image_name"]))
    #             txt_file.write("Inference time: {:.2f} seconds\n".format(result["inference_time"]))
    #             txt_file.write("Detections:\n")

    #             # Список всех детекций
    #             if result["detections"]:
    #                 for detection in result["detections"]:
    #                     bbox = detection["bbox"]
    #                     class_id = detection["class_id"]
    #                     score = detection["score"]
    #                     txt_file.write(
    #                         "  - Class ID: {}, Score: {:.2f}, BBox: [x={}, y={}, w={}, h={}]\n".format(
    #                             class_id, score, bbox[0], bbox[1], bbox[2], bbox[3]
    #                         )
    #                     )
    #             else:
    #                 txt_file.write("  No detections.\n")
    #             txt_file.write("\n")  # Разделение между изображениями

    #         print("Все результаты сохранены в", output_txt_path)
    # except OSError as e:
    #     print("Ошибка сохранения текстового файла:", e)

    for result in results:
        # Заголовок для каждого изображения
        print("Image: {}".format(result["image_name"]))
        print("Inference time: {:.2f} seconds".format(result["inference_time"]))
        print("Detections:")

        # Список всех детекций
        if result["detections"]:
            for detection in result["detections"]:
                bbox = detection["bbox"]
                class_id = detection["class_id"]
                score = detection["score"]
                print(
                    "  - Class ID: {}, Score: {:.2f}, BBox: [x={}, y={}, w={}, h={}]".format(
                        class_id, score, bbox[0], bbox[1], bbox[2], bbox[3]
                    )
                )
        else:
            print("  No detections.")
        print("\n")  # Разделение между изображениями


