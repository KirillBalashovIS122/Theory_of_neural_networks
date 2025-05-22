# model.py
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import re
from tensorflow.keras import layers, Model, optimizers, losses

class SegmentationModel:
    def __init__(self):
        self.model = None
        self.last_mask = None
        self.CLASS_NAMES = {
            0: "Фон",
            1: "Кошка",
            2: "Собака"
        }
        self.CLASS_COLORS = {
            0: [0, 0, 0],    # Черный
            1: [255, 0, 0],  # Красный
            2: [0, 0, 255]   # Синий
        }
        self.input_size = 256
        self.num_classes = 3

    def load_pretrained_model(self):
        try:
            model_url = "https://tfhub.dev/tensorflow/deeplabv3/mobilenet_v2/1"
            self.model = hub.load(model_url)
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def create_custom_model(self):
        inputs = layers.Input(shape=(self.input_size, self.input_size, 3))
        
        # Энкодер
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D()(x)
        
        # Ботлнек
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        
        # Декодер
        x = layers.UpSampling2D()(x)
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
        
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )

    def train_custom_model(self, dataset_path):
        self.create_custom_model()
        train_dataset, val_dataset = self.prepare_data(dataset_path)
        
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=10,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3),
                tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
            ]
        )

    def prepare_data(self, dataset_path):
        def process_path(img_path, mask_path):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [self.input_size, self.input_size])
            img = tf.cast(img, tf.float32) / 255.0
            
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=1)
            mask = tf.image.resize(mask, [self.input_size, self.input_size])
            mask = tf.cast(mask, tf.int32)
            
            return img, mask

        img_dir = os.path.join(dataset_path, 'images')
        mask_dir = os.path.join(dataset_path, 'masks')
        
        img_paths = [os.path.join(img_dir, f) for f in sorted(os.listdir(img_dir))]
        mask_paths = [os.path.join(mask_dir, f) for f in sorted(os.listdir(mask_dir))]
        
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
        dataset = dataset.map(process_path).batch(8).prefetch(2)
        
        train_size = int(0.8 * len(dataset))
        return dataset.take(train_size), dataset.skip(train_size)

    def predict(self, image, selected_classes, transparency):
        if self.model is None:
            raise ValueError("Модель не инициализирована")
        
        # Предобработка
        input_img = cv2.resize(image, (self.input_size, self.input_size))
        input_tensor = tf.expand_dims(input_img, 0)
        
        # Предсказание
        if isinstance(self.model, tf.keras.Model):
            pred = self.model.predict(input_tensor)[0]
            mask = np.argmax(pred, axis=-1)
        else:
            output = self.model(input_tensor)
            mask = output['semantic_predictions'][0].numpy()
        
        self.last_mask = mask
        return self.apply_segmentation(image, mask, selected_classes, transparency)

    def apply_segmentation(self, image, mask, classes, transparency):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id in classes:
            colored_mask[mask == class_id] = self.CLASS_COLORS.get(class_id, [0,0,0])
        
        return cv2.addWeighted(image, 1 - transparency, colored_mask, transparency, 0)

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def is_hex_color(self, color):
        return bool(re.match(r'^#(?:[0-9a-fA-F]{3}){1,2}$', color))