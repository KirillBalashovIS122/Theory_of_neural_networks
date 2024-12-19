import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from collections import Counter
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(path):
    """
    Загружает и предварительно обрабатывает изображение.
    
    Args:
        path (str): Путь к изображению.
    
    Returns:
        tf.Tensor: Нормализованное изображение или None, если загрузка не удалась.
    """
    try:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_height, img_width])
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Ошибка при загрузке изображения {path}: {e}")
        return None

def load_dataset(data_dir):
    """
    Рекурсивно обходит директорию и загружает изображения.
    
    Args:
        data_dir (str): Путь к директории с изображениями.
    
    Returns:
        list: Список путей к изображениям.
        list: Список меток.
        list: Список имен классов.
    """
    image_paths = []
    labels = []
    class_names = set()
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    rel_path = os.path.relpath(file_path, data_dir)
                    class_name = os.path.basename(os.path.dirname(rel_path))
                    class_names.add(class_name)
                    image_paths.append(file_path)  # Убедимся, что это строка
                    labels.append(class_name)
                except Exception as e:
                    print(f"Ошибка при загрузке изображения {file_path}: {e}")
                    os.remove(file_path)
    
    class_names = sorted(list(class_names))
    label_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    labels = [label_to_index[label] for label in labels]
    
    # Проверка типов данных
    if not all(isinstance(path, str) for path in image_paths):
        raise TypeError("Все пути к изображениям должны быть строками.")
    
    return image_paths, labels, class_names

data_dir = '/home/kbalashov/VS_Code/TONN/LB_4/sources/seasons_1'
batch_size = 32
img_height = 150
img_width = 150

print("Загрузка данных из директории...")
image_paths, labels, class_names = load_dataset(data_dir)

print(f"Общее количество изображений: {len(image_paths)}")

# Разделение на обучающую и валидационную выборки с сохранением баланса классов
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

# Проверка уникальных меток в обучающей и валидационной выборках
unique_train_labels = np.unique(train_labels)
unique_val_labels = np.unique(val_labels)

if len(unique_train_labels) < len(class_names):
    print(f"Внимание: В обучающей выборке отсутствуют метки: {set(class_names) - set(unique_train_labels)}")
if len(unique_val_labels) < len(class_names):
    print(f"Внимание: В валидационной выборке отсутствуют метки: {set(class_names) - set(unique_val_labels)}")

# Создание tf.data.Dataset для обучающей и валидационной выборок
train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

train_ds = train_ds.map(lambda path, label: (load_and_preprocess_image(path), label)).batch(batch_size)
val_ds = val_ds.map(lambda path, label: (load_and_preprocess_image(path), label)).batch(batch_size)

print(f"Количество изображений в обучающей выборке: {len(train_labels)}")
print(f"Количество изображений в валидационной выборке: {len(val_labels)}")

# Проверка баланса данных
train_labels_array = np.array(train_labels)
val_labels_array = np.array(val_labels)

print(f"Распределение меток в обучающей выборке: {Counter(train_labels_array)}")
print(f"Распределение меток в валидационной выборке: {Counter(val_labels_array)}")

num_classes = len(class_names)

# Создание первой модели с одним сверточным слоем и одним Dense слоем
print("Создание первой модели с одним сверточным слоем и одним Dense слоем...")
model1 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(num_classes, activation='softmax')
])

model1.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Создание второй модели (оставляем без изменений)
print("Создание второй модели сверточной нейронной сети...")
model2 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Вывод сводки моделей
print("Сводка первой модели:")
model1.summary()
print("Сводка второй модели:")
model2.summary()

# Обучение первой модели
print("Обучение первой модели...")
history1 = model1.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Обучение второй модели
print("Обучение второй модели...")
history2 = model2.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Визуализация кривых обучения для первой модели
acc1 = history1.history['accuracy']
if 'val_accuracy' in history1.history:
    val_acc1 = history1.history['val_accuracy']
else:
    val_acc1 = None
    print("Валидационная выборка пуста, проверьте разделение данных.")

loss1 = history1.history['loss']
if 'val_loss' in history1.history:
    val_loss1 = history1.history['val_loss']
else:
    val_loss1 = None

epochs_range = range(10)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc1, label='Точность обучения (Model 1)')
if val_acc1 is not None:
    plt.plot(epochs_range, val_acc1, label='Точность валидации (Model 1)')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации (Model 1)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss1, label='Потери обучения (Model 1)')
if val_loss1 is not None:
    plt.plot(epochs_range, val_loss1, label='Потери валидации (Model 1)')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации (Model 1)')
plt.show()

# Визуализация кривых обучения для второй модели
acc2 = history2.history['accuracy']
if 'val_accuracy' in history2.history:
    val_acc2 = history2.history['val_accuracy']
else:
    val_acc2 = None
    print("Валидационная выборка пуста, проверьте разделение данных.")

loss2 = history2.history['loss']
if 'val_loss' in history2.history:
    val_loss2 = history2.history['val_loss']
else:
    val_loss2 = None

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc2, label='Точность обучения (Model 2)')
if val_acc2 is not None:
    plt.plot(epochs_range, val_acc2, label='Точность валидации (Model 2)')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации (Model 2)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss2, label='Потери обучения (Model 2)')
if val_loss2 is not None:
    plt.plot(epochs_range, val_loss2, label='Потери валидации (Model 2)')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации (Model 2)')
plt.show()

# Тестирование первой модели
print("Тестирование первой модели...")
if len(val_ds) > 0:
    test_loss1, test_acc1 = model1.evaluate(val_ds)
    print(f'Точность на тестовых данных (Model 1): {test_acc1}')
else:
    print("Валидационная выборка пуста, пропустите тестирование.")

# Тестирование второй модели
print("Тестирование второй модели...")
if len(val_ds) > 0:
    test_loss2, test_acc2 = model2.evaluate(val_ds)
    print(f'Точность на тестовых данных (Model 2): {test_acc2}')
else:
    print("Валидационная выборка пуста, пропустите тестирование.")

# Оценка первой модели
if len(val_ds) > 0:
    y_pred1 = model1.predict(val_ds)
    y_pred_classes1 = np.argmax(y_pred1, axis=1)
    y_true = np.concatenate([y for x, y in val_ds], axis=0)

    # Проверка уникальных меток
    print(f"Уникальные метки в y_true: {np.unique(y_true)}")
    print(f"Уникальные метки в y_pred_classes1: {np.unique(y_pred_classes1)}")

    print("Отчет о классификации для первой модели:")
    print(classification_report(y_true, y_pred_classes1, target_names=class_names, zero_division=0))

    print("Матрица ошибок для первой модели:")
    conf_matrix1 = confusion_matrix(y_true, y_pred_classes1)
    print(conf_matrix1)

    # Вычисление ROC-кривой для первой модели
    fpr1, tpr1, thresholds1 = roc_curve(y_true, y_pred1[:, 1], pos_label=1)
    roc_auc1 = auc(fpr1, tpr1)

    # Построение ROC-кривой для первой модели
    plt.figure()
    plt.plot(fpr1, tpr1, color='darkorange', lw=2, label='ROC кривая (Model 1, площадь = %0.2f)' % roc_auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Model 1)')
    plt.legend(loc="lower right")
    plt.show()

# Оценка второй модели
if len(val_ds) > 0:
    y_pred2 = model2.predict(val_ds)
    y_pred_classes2 = np.argmax(y_pred2, axis=1)

    # Проверка уникальных меток
    print(f"Уникальные метки в y_true: {np.unique(y_true)}")
    print(f"Уникальные метки в y_pred_classes2: {np.unique(y_pred_classes2)}")

    print("Отчет о классификации для второй модели:")
    print(classification_report(y_true, y_pred_classes2, target_names=class_names, zero_division=0))

    print("Матрица ошибок для второй модели:")
    conf_matrix2 = confusion_matrix(y_true, y_pred_classes2)
    print(conf_matrix2)

    # Вычисление ROC-кривой для второй модели
    fpr2, tpr2, thresholds2 = roc_curve(y_true, y_pred2[:, 1], pos_label=1)
    roc_auc2 = auc(fpr2, tpr2)

    # Построение ROC-кривой для второй модели
    plt.figure()
    plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='ROC кривая (Model 2, площадь = %0.2f)' % roc_auc2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Model 2)')
    plt.legend(loc="lower right")
    plt.show()

# Выводы
print("Выводы:")
if len(val_ds) > 0:
    print(f"Точность первой модели: {test_acc1}")
    print(f"Точность второй модели: {test_acc2}")

    if test_acc1 > test_acc2:
        print("Первая модель показала лучшие результаты.")
    else:
        print("Вторая модель показала лучшие результаты.")
else:
    print("Валидационная выборка пуста, невозможно сравнить модели.")
