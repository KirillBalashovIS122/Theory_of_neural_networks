import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def load_and_preprocess_image(path):
    """
    Загружает и предварительно обрабатывает изображение.
    
    Args:
        path (str): Путь к изображению.
    
    Returns:
        tf.Tensor: Нормализованное изображение.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0
    return img

def load_dataset(data_dir):
    """
    Рекурсивно обходит директорию и загружает изображения.
    
    Args:
        data_dir (str): Путь к директории с изображениями.
    
    Returns:
        tf.data.Dataset: Датасет с изображениями и метками.
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
                    image_paths.append(file_path)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Ошибка при загрузке изображения {file_path}: {e}")
                    os.remove(file_path)
    
    class_names = sorted(list(class_names))
    label_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}
    labels = [label_to_index[label] for label in labels]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda path, label: (load_and_preprocess_image(path), label))
    return dataset, class_names

data_dir = '/home/kbalashov/VS_Code/TONN/LB_3/sources/seasons'
batch_size = 32
img_height = 150
img_width = 150

print("Загрузка данных из директории...")
dataset, class_names = load_dataset(data_dir)

# Проверка общего количества изображений
print(f"Общее количество изображений: {len(dataset)}")

# Разделение на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))
train_ds = dataset.take(train_size).batch(batch_size, drop_remainder=True)
val_ds = dataset.skip(train_size).batch(batch_size, drop_remainder=True)

# Проверка количества изображений в обучающей и валидационной выборках
print(f"Количество изображений в обучающей выборке: {len(list(train_ds)) * batch_size}")
print(f"Количество изображений в валидационной выборке: {len(list(val_ds)) * batch_size}")

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

print("Обучение модели...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

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

print("Тестирование модели...")
test_loss, test_acc = model.evaluate(val_ds)
print(f'Точность на тестовых данных: {test_acc}')

y_pred = model.predict(val_ds)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.concatenate([y for x, y in val_ds], axis=0)

# Проверка количества изображений в тестовой выборке
print(f"Количество изображений в тестовой выборке: {len(y_true)}")

print("Отчет о классификации:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("Матрица ошибок:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Проверка количества изображений в матрице ошибок
print(f"Количество изображений в матрице ошибок: {conf_matrix.sum()}")

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

# Отображение верно и неверно классифицированных изображений в одной фигуре
correct_indices = np.where(y_pred_classes == y_true)[0]
incorrect_indices = np.where(y_pred_classes != y_true)[0]

# Создаем сетку для отображения изображений
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Классифицированные изображения')

# Отображение верно классифицированных изображений
for i, ax in enumerate(axes[0]):
    if i < len(correct_indices):
        img, label = next(iter(val_ds.unbatch().skip(correct_indices[i]).batch(1)))
        img = img[0].numpy()
        img = (img * 255).astype("uint8")
        ax.imshow(img)
        ax.set_title(f'Правильный: {class_names[label.numpy()[0]]}\nПредсказано: {class_names[y_pred_classes[correct_indices[i]]]}')
        ax.axis('off')
    else:
        ax.axis('off')

# Отображение неверно классифицированных изображений
for i, ax in enumerate(axes[1]):
    if i < len(incorrect_indices):
        img, label = next(iter(val_ds.unbatch().skip(incorrect_indices[i]).batch(1)))
        img = img[0].numpy()
        img = (img * 255).astype("uint8")
        ax.imshow(img)
        ax.set_title(f'Правильный: {class_names[label.numpy()[0]]}\nПредсказано: {class_names[y_pred_classes[incorrect_indices[i]]]}')
        ax.axis('off')
    else:
        ax.axis('off')

plt.tight_layout()
plt.show()
