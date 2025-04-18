import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SegmentationModel:
    def __init__(self, base_dir, image_size=(256, 256), batch_size=4, classes=None):
        self.BASE_DIR = base_dir
        self.MODELS_DIR = os.path.join(base_dir, 'models')
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        self.IMAGE_SIZE = image_size
        self.BATCH_SIZE = batch_size
        self.CLASSES = classes or {
            'cloudy': 0,
            'desert': 1,
            'green_area': 2,
            'water': 3
        }
        self.model = None

    def load_and_preprocess_data(self, data_path):
        image_paths = []
        mask_values = []
        
        if not os.path.exists(data_path):
            raise ValueError(f"Директория данных не найдена: {data_path}")

        for label in self.CLASSES:
            class_path = os.path.join(data_path, label)
            if not os.path.exists(class_path):
                raise ValueError(f"Директория класса {label} не найдена: {class_path}")

            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                raise ValueError(f"Нет изображений в директории: {class_path}")

            for img_name in images:
                image_paths.append(os.path.join(class_path, img_name))
                mask_values.append(self.CLASSES[label])

        if not image_paths:
            raise ValueError("Не найдено изображений для обучения")

        buffer_size = max(1000, len(image_paths))

        def _process_path(img_path, mask_val):
            img = tf.io.read_file(img_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, self.IMAGE_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            mask = tf.zeros(self.IMAGE_SIZE + (1,), dtype=tf.uint8) + tf.cast(mask_val, tf.uint8)
            return img, mask

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_values))
        dataset = dataset.shuffle(buffer_size).map(_process_path, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def augment_data(self, image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        angle = tf.random.uniform([], -0.2, 0.2)
        k = tf.cast(angle * 180 / np.pi, tf.int32) // 90
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, mask

    def build_unet(self, input_size=(256, 256, 3)):
        inputs = Input(input_size)
        
        c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = BatchNormalization()(c2)
        c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = BatchNormalization()(c3)
        c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
        
        u4 = UpSampling2D((2, 2))(c3)
        u4 = Conv2D(64, 2, activation='relu', padding='same')(u4)
        u4 = concatenate([u4, c2])
        c4 = Conv2D(64, 3, activation='relu', padding='same')(u4)
        c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)
        
        u5 = UpSampling2D((2, 2))(c4)
        u5 = Conv2D(32, 2, activation='relu', padding='same')(u5)
        u5 = concatenate([u5, c1])
        c5 = Conv2D(32, 3, activation='relu', padding='same')(u5)
        c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)
        
        outputs = Conv2D(4, 1, activation='softmax', dtype='float32')(c5)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return self.model

    def train(self, data_path, epochs=5):
        dataset = self.load_and_preprocess_data(data_path)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
        train_size = int(0.8 * len(list(dataset)))
        train_dataset = dataset.take(train_size).map(
            self.augment_data,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_dataset = dataset.skip(train_size)
        
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.MODELS_DIR, 'final_model.keras'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max'
            ),
            EarlyStopping(patience=5, restore_best_weights=True)
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        final_model_path = os.path.join(self.MODELS_DIR, 'final_model.keras')
        self.model.save(final_model_path)
        return history

    def load_saved_model(self, model_name='final_model.keras'):
        model_path = os.path.join(self.MODELS_DIR, model_name)
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            return True
        return False
