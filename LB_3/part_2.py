import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Загрузка данных с использованием tf.data.Dataset
data_dir = '/home/kbalashov/VS_Code/TONN/LB_3/sources/seasons'
batch_size = 32
img_height = 150
img_width = 150

# Функция для загрузки и предварительной обработки изображений
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Нормализация
    return img

# Функция для рекурсивного обхода папок и загрузки изображений
def load_dataset(data_dir):
    image_paths = []
    labels = []
    class_names = set()  # Используем множество для хранения уникальных имен классов
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    # Получаем относительный путь от корневой директории
                    rel_path = os.path.relpath(file_path, data_dir)
                    # Определяем класс как имя папки, в которой находится изображение
                    class_name = os.path.basename(os.path.dirname(rel_path))
                    class_names.add(class_name)  # Добавляем класс в множество
                    image_paths.append(file_path)
                    labels.append(class_name)  # Используем имя класса как метку
                except Exception as e:
                    print(f"Ошибка при загрузке изображения {file_path}: {e}")
                    os.remove(file_path)  # Удаление поврежденного изображения
            else:
                print(f"Пропускаем некорректный файл или директорию: {file_path}")
    
    # Преобразуем множество имен классов в список и сортируем
    class_names = sorted(list(class_names))
    
    # Преобразуем метки в индексы классов
    label_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    labels = [label_to_index[label] for label in labels]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda path, label: (load_and_preprocess_image(path), label))
    return dataset, class_names

print("Загрузка данных из директории...")
dataset, class_names = load_dataset(data_dir)

# Разделение на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size).batch(batch_size)
val_ds = dataset.skip(train_size).batch(batch_size)

# Создание модели многослойного персептрона (MLP)
num_classes = len(class_names)

print("Создание модели многослойного персептрона...")
model = Sequential([
    Flatten(input_shape=(img_height, img_width, 3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
print("Обучение модели...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# Отображение графиков зависимости loss и accuracy
print("Отображение графиков зависимости loss и accuracy...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

# Вывод информации о графиках в консоль
print("\nИнформация о графиках:")
print(f"Средняя точность обучения: {np.mean(acc):.4f}")
print(f"Средняя точность валидации: {np.mean(val_acc):.4f}")
print(f"Средние потери обучения: {np.mean(loss):.4f}")
print(f"Средние потери валидации: {np.mean(val_loss):.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Точность обучения')
plt.plot(epochs_range, val_acc, label='Точность валидации')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Потери обучения')
plt.plot(epochs_range, val_loss, label='Потери валидации')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации')
plt.show()

# Тестирование модели
print("Тестирование модели...")
test_loss, test_acc = model.evaluate(val_ds)
print(f'Точность на тестовых данных: {test_acc}')

# Оценка точности, полноты, аккуратности, F1-меры, матрицы ошибок и ROC кривой
print("Оценка точности модели...")
y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)

print("Отчет о классификации:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("Матрица ошибок:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# ROC кривая
fpr, tpr, thresholds = roc_curve(y_true, y_pred.max(axis=1))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC кривая (площадь = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Отображение верно и неверно классифицированных изображений
print("Отображение верно классифицированных изображений...")
correct_indices = np.where(y_pred_classes == y_true)[0]
for i in range(min(3, len(correct_indices))):
    img, label = next(iter(val_ds.unbatch().skip(correct_indices[i]).batch(1)))
    img = img[0].numpy()  # Извлекаем изображение из батча
    img = (img * 255).astype("uint8")  # Возвращаем изображение в исходный диапазон
    plt.imshow(img)
    plt.title(f'Правильный: {class_names[label.numpy()[0]]}, Предсказано: {class_names[y_pred_classes[correct_indices[i]]]}')
    plt.show()

print("Отображение неверно классифицированных изображений...")
incorrect_indices = np.where(y_pred_classes != y_true)[0]
for i in range(min(3, len(incorrect_indices))):
    img, label = next(iter(val_ds.unbatch().skip(incorrect_indices[i]).batch(1)))
    img = img[0].numpy()  # Извлекаем изображение из батча
    img = (img * 255).astype("uint8")  # Возвращаем изображение в исходный диапазон
    plt.imshow(img)
    plt.title(f'Правильный: {class_names[label.numpy()[0]]}, Предсказано: {class_names[y_pred_classes[incorrect_indices[i]]]}')
    plt.show()
