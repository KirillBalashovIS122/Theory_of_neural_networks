import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Параметры
data_dir = '/home/kbalashov/VS_Code/TONN/LB_4/sources/seasons_1'
batch_size = 32
img_height = 150
img_width = 150
num_classes = 2  # Количество классов в датасете (2 класса: summer и winter)

# Загрузка данных
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Используем 'binary' для бинарной классификации
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Используем 'binary' для бинарной классификации
    subset='validation'
)

# Загрузка базовой модели VGG16 без верхнего слоя классификатора
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Заморозка базовых слоев
for layer in base_model.layers:
    layer.trainable = False

# Добавление новых слоев
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)  # Используем 'sigmoid' для бинарной классификации

# Создание модели
model = Model(inputs=base_model.input, outputs=predictions)

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',  # Используем 'binary_crossentropy' для бинарной классификации
              metrics=['accuracy'])

# Обучение модели
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Визуализация кривых обучения
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

# Тестирование модели
test_loss, test_acc = model.evaluate(val_generator)
print(f'Точность на тестовых данных: {test_acc}')

# Оценка модели
y_pred = model.predict(val_generator)
y_pred_classes = (y_pred > 0.5).astype(int)  # Преобразуем вероятности в бинарные метки
y_true = val_generator.classes

print("Отчет о классификации:")
print(classification_report(y_true, y_pred_classes, target_names=val_generator.class_indices))

print("Матрица ошибок:")
conf_matrix = confusion_matrix(y_true, y_pred_classes)
print(conf_matrix)

# Вычисление ROC-кривой
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Построение ROC-кривой
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC кривая (площадь = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Файнтюнинг (если необходимо)
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

# Визуализация кривых обучения после файнтюнинга
acc_finetune = history_finetune.history['accuracy']
val_acc_finetune = history_finetune.history['val_accuracy']
loss_finetune = history_finetune.history['loss']
val_loss_finetune = history_finetune.history['val_loss']

epochs_range_finetune = range(5)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range_finetune, acc_finetune, label='Точность обучения (файнтюнинг)')
plt.plot(epochs_range_finetune, val_acc_finetune, label='Точность валидации (файнтюнинг)')
plt.legend(loc='lower right')
plt.title('Точность обучения и валидации (файнтюнинг)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range_finetune, loss_finetune, label='Потери обучения (файнтюнинг)')
plt.plot(epochs_range_finetune, val_loss_finetune, label='Потери валидации (файнтюнинг)')
plt.legend(loc='upper right')
plt.title('Потери обучения и валидации (файнтюнинг)')
plt.show()

# Тестирование модели после файнтюнинга
test_loss_finetune, test_acc_finetune = model.evaluate(val_generator)
print(f'Точность на тестовых данных после файнтюнинга: {test_acc_finetune}')
