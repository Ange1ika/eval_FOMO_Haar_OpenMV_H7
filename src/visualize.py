import matplotlib.pyplot as plt
import numpy as np
print(plt.style.available)
# Установим стиль шрифта и размер
plt.style.use("fivethirtyeight") 


plt.rc('font', family='serif', size=12)

models = ['YOLOv8n', 'YOLOv8s', 'FOMO', 'Haar']
mAP50_95 = [0.5979, 0.7111, 0.4874, 0.16]

# Первый график: mAP
x = np.arange(len(models))
width = 0.4

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(
    x, 
    mAP50_95, 
    width, 
    label='mAP50-95', 
    color='skyblue', 
    edgecolor='black', 
    alpha=0.8
)

ax.set_ylabel('Значение метрики', fontsize=14)
ax.set_title('Сравнение mAP', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, rotation=15)
ax.legend(fontsize=12)

# Добавим значения над колонками
for rect in rects1:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, height + 0.02, f'{height:.2f}', ha='center', fontsize=10)

fig.tight_layout()
plt.show()

# Второй график: Precision и Recall
precision_0_5 = [0.628, 0.731, 0.5556, 0.59]
recall_0_5 = [0.833, 0.85, 0.20, 0.10]

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(
    x - width / 2, 
    precision_0_5, 
    width, 
    label='Precision_0.5', 
    color='lightcoral', 
    edgecolor='black', 
    alpha=0.8
)
rects2 = ax.bar(
    x + width / 2, 
    recall_0_5, 
    width, 
    label='Recall_0.5', 
    color='limegreen', 
    edgecolor='black', 
    alpha=0.8
)

ax.set_ylabel('Значение метрик', fontsize=14)
ax.set_title('Сравнение точности и полноты при пороге IOU 0.5', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, rotation=15)
ax.legend(fontsize=12)

# Добавим значения над колонками
for rect in rects1 + rects2:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, height + 0.02, f'{height:.2f}', ha='center', fontsize=10)

fig.tight_layout()
plt.show()

# Третий график: Время инференса
inference_time = [1993, 2397.7, 80, 15]

fig, ax = plt.subplots(figsize=(8, 5))
rects = ax.bar(
    models, 
    inference_time, 
    color=['blue', 'green', 'red', 'purple'], 
    edgecolor='black', 
    alpha=0.8
)

ax.set_ylabel('Время инференса (ms)', fontsize=14)
ax.set_title('Сравнение производительности моделей', fontsize=16)
ax.set_yscale('log')  # Логарифмическая шкала для улучшенного восприятия

# Добавим значения над колонками
for rect in rects:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2.0, height + 10, f'{height:.1f}', ha='center', fontsize=10)

fig.tight_layout()
plt.show()

# Четвертый график: Точность и полнота
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, precision_0_5, label='Точность', color='blue', s=100)
ax.scatter(x, recall_0_5, label='Полнота', color='red', s=100)

# Аннотации для точек
for i, model in enumerate(models):
    ax.annotate(f'{precision_0_5[i]:.2f}', (x[i], precision_0_5[i] + 0.02), ha='center', fontsize=10, color='blue')
    ax.annotate(f'{recall_0_5[i]:.2f}', (x[i], recall_0_5[i] - 0.05), ha='center', fontsize=10, color='red')

ax.set_ylabel('Значение метрик', fontsize=14)
ax.set_title('Сравнение качественных параметров', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, rotation=15)
ax.legend(fontsize=12)

fig.tight_layout()
plt.show()
