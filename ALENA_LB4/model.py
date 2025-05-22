import tensorflow as tf
import cv2
import numpy as np
import os
import re
import random
import time
from tensorflow.keras import layers, Model, optimizers, losses
from sklearn.cluster import KMeans

class SegmentationModel:
    def __init__(self):
        self.model = None
        self.last_mask = None
        self.CLASS_NAMES = {
            0: "Фон",
            1: "Животное"
        }
        self.CLASS_COLORS = {
            0: [0, 0, 0],
            1: [255, 0, 0]
        }
        self.input_size = 256
        self.num_classes = 1
        self.max_images = 1000

    def load_pretrained_model(self):
        try:
            self.create_custom_model()
            print("Модель успешно создана")
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            return False

    def create_custom_model(self):
        try:
            inputs = layers.Input(shape=(self.input_size, self.input_size, 3))
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x1 = layers.MaxPooling2D()(x)

            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x1)
            x = layers.BatchNormalization()(x)
            x2 = layers.MaxPooling2D()(x)

            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x2)
            x = layers.BatchNormalization()(x)
            
            x = layers.UpSampling2D()(x)
            x = layers.Concatenate()([x, x1])
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            outputs = layers.Conv2D(self.num_classes, 1, activation='sigmoid')(x)
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(
                optimizer=optimizers.Adam(),
                loss=losses.BinaryCrossentropy(),
                metrics=['accuracy']
            )
            print("Модель успешно скомпилирована")
        except Exception as e:
            print(f"Ошибка создания модели: {str(e)}")
            raise

    def train_custom_model(self, dataset_path):
        try:
            start_time = time.time()
            print("Начало обучения модели...")
            self.create_custom_model()
            train_dataset, val_dataset, train_steps, val_steps = self.prepare_data(dataset_path)
            if train_steps == 0:
                raise ValueError("Обучающая выборка пуста")
            if val_steps == 0:
                print("Предупреждение: Валидационная выборка пуста, обучение без валидации")
            history = self.model.fit(
                train_dataset,
                validation_data=val_dataset if val_steps > 0 else None,
                epochs=5,
                steps_per_epoch=train_steps,
                validation_steps=val_steps if val_steps > 0 else None,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss', restore_best_weights=True),
                    tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')
                ]
            )
            end_time = time.time()
            print(f"Обучение завершено за {end_time - start_time:.2f} секунд")
            return history
        except Exception as e:
            print(f"Ошибка обучения модели: {str(e)}")
            raise

    def prepare_data(self, dataset_path):
        try:
            def process_image(img_path):
                try:
                    img = tf.io.read_file(img_path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    img = tf.image.resize(img, [self.input_size, self.input_size])
                    img = tf.cast(img, tf.float32) / 255.0
                    img_np = img.numpy()
                    pixels = img_np.reshape(-1, 3)
                    kmeans = KMeans(n_clusters=2, random_state=0, n_init=1).fit(pixels)
                    labels = kmeans.labels_.reshape(self.input_size, self.input_size)
                    cluster_means = [np.mean(pixels[labels.flatten() == i]) for i in range(2)]
                    bg_cluster = np.argmax(cluster_means)
                    labels = (labels != bg_cluster).astype(np.float32)
                    labels = labels[..., np.newaxis]
                    return img, labels
                except Exception as e:
                    print(f"Ошибка обработки изображения {img_path}: {str(e)}")
                    raise

            def tf_process_image(img_path):
                try:
                    img, labels = tf.py_function(process_image, [img_path], [tf.float32, tf.float32])
                    img.set_shape([self.input_size, self.input_size, 3])
                    labels.set_shape([self.input_size, self.input_size, 1])
                    return img, labels
                except Exception as e:
                    print(f"Ошибка TensorFlow обработки изображения: {str(e)}")
                    raise

            img_paths = [os.path.join(dataset_path, f) for f in sorted(os.listdir(dataset_path))
                        if f.endswith(('.jpg', '.jpeg', '.png'))]
            if len(img_paths) > self.max_images:
                img_paths = random.sample(img_paths, self.max_images)
                print(f"Выбрано {self.max_images} случайных изображений")
            else:
                print(f"Используется {len(img_paths)} изображений")
            dataset = tf.data.Dataset.from_tensor_slices(img_paths)
            dataset = dataset.map(tf_process_image, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.batch(4).prefetch(2).repeat()
            train_size = int(0.8 * len(img_paths))
            val_size = len(img_paths) - train_size
            train_steps = max(1, train_size // 4)
            val_steps = max(1, val_size // 4) if val_size > 0 else 0
            train_dataset = dataset.take(train_steps)
            val_dataset = dataset.skip(train_steps).take(val_steps) if val_steps > 0 else dataset.take(0)
            return train_dataset, val_dataset, train_steps, val_steps
        except Exception as e:
            print(f"Ошибка подготовки данных: {str(e)}")
            raise

    def predict(self, image, selected_classes, transparency):
        try:
            if self.model is None:
                print("Ошибка: Модель не инициализирована")
                raise ValueError("Модель не инициализирована")
            input_img = cv2.resize(image, (self.input_size, self.input_size))
            input_img = input_img.astype(np.float32) / 255.0
            input_tensor = tf.expand_dims(input_img, 0)
            pred = self.model.predict(input_tensor)[0]
            mask = (pred > 0.5).astype(np.uint8)
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            self.last_mask = mask
            return self.apply_segmentation(image, mask, selected_classes, transparency)
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            raise

    def apply_segmentation(self, image, mask, classes, transparency):
        try:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for class_id in classes:
                colored_mask[mask == class_id] = self.CLASS_COLORS.get(class_id, [0,0,0])
            return cv2.addWeighted(image, 1 - transparency, colored_mask, transparency, 0)
        except Exception as e:
            print(f"Ошибка применения сегментации: {str(e)}")
            raise

    def rgb_to_hex(self, rgb):
        try:
            return "#{:02x}{:02x}{:02x}".format(*rgb)
        except Exception as e:
            print(f"Ошибка конвертации RGB в HEX: {str(e)}")
            raise

    def hex_to_rgb(self, hex_color):
        try:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except Exception as e:
            print(f"Ошибка конвертации HEX в RGB: {str(e)}")
            raise

    def is_hex_color(self, color):
        try:
            return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color))
        except Exception as e:
            print(f"Ошибка проверки HEX-цвета: {str(e)}")
            raise