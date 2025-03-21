"""
Модуль обработки данных и управления моделями.
Содержит логику работы с данными и обучения моделей.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.callbacks import LambdaCallback
from skimage.metrics import peak_signal_noise_ratio as psnr
from models import DenseAutoencoder, ConvAutoencoder
import logging

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class ImageDenoiser:
    """Класс для управления процессом шумоподавления"""
    def __init__(self, noise_factor=0.3, epochs=8):
        """
        Инициализация обработчика
        Args:
            noise_factor (float): Уровень добавляемого шума (0.1-0.5)
            epochs (int): Количество эпох обучения
        """
        self.noise_factor = noise_factor
        self.epochs = epochs
        self.models = {'dense': None, 'conv': None}
        self.models_trained = False
        self.data_loaded = False

    def load_data(self):
        """Загрузка и подготовка набора данных Fashion MNIST"""
        logging.info("Загрузка данных Fashion MNIST...")
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
        
        logging.info("Нормализация данных...")
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        
        self._update_noisy_data()
        self.data_loaded = True
        logging.info(f"Данные подготовлены с фактором шума {self.noise_factor}")

    def _update_noisy_data(self):
        """Обновление зашумленных версий данных"""
        self.x_train_noisy = self._add_noise(self.x_train)
        self.x_test_noisy = self._add_noise(self.x_test)
        self.x_train_conv = np.expand_dims(self.x_train, -1)
        self.x_test_conv = np.expand_dims(self.x_test, -1)
        self.x_train_noisy_conv = np.expand_dims(self.x_train_noisy, -1)
        self.x_test_noisy_conv = np.expand_dims(self.x_test_noisy, -1)

    def update_noise(self):
        """Обновление уровня шума в данных"""
        self._update_noisy_data()

    def _add_noise(self, data):
        """
        Добавление гауссовского шума к данным
        Args:
            data (np.array): Исходные изображения
        Returns:
            np.array: Зашумленные изображения
        """
        noisy = data + self.noise_factor * np.random.normal(size=data.shape)
        return np.clip(noisy, 0., 1.)

    def build_models(self):
        """Инициализация моделей автоэнкодеров"""
        logging.info("Инициализация моделей автоэнкодеров...")
        self.models['dense'] = DenseAutoencoder()
        self.models['conv'] = ConvAutoencoder()

    def train_models(self, progress_callback=None):
        """
        Обучение моделей
        Args:
            progress_callback (function): Функция для отслеживания прогресса
        """
        logging.info("Начато обучение полносвязной модели")
        self._train_single_model('dense', progress_callback)
        
        logging.info("Начато обучение сверточной модели")
        self._train_single_model('conv', progress_callback)
        
        self.models_trained = True
        logging.info("Обучение всех моделей завершено")

    def _train_single_model(self, model_type, progress_callback=None):
        """
        Обучение одной модели
        Args:
            model_type (str): Тип модели ('dense' или 'conv')
            progress_callback (function): Коллбэк для обновления прогресса
        """
        model = self.models[model_type]
        model.compile(optimizer='adam', loss=losses.MeanSquaredError())
        
        train_data = self.x_train_noisy if model_type == 'dense' else self.x_train_noisy_conv
        val_data = self.x_test_noisy if model_type == 'dense' else self.x_test_noisy_conv
        
        model.fit(
            train_data,
            self.x_train if model_type == 'dense' else self.x_train_conv,
            epochs=self.epochs,
            batch_size=512,
            verbose=0,
            validation_data=(val_data, 
                            self.x_test if model_type == 'dense' else self.x_test_conv),
            callbacks=[LambdaCallback(on_epoch_end=lambda epoch, logs: 
                                     progress_callback(epoch, logs)) 
                      ] if progress_callback else None
        )

    def evaluate_models(self):
        logging.info("Начата оценка моделей")
        try:
            if not self.data_loaded:
                logging.error("Данные не загружены")
                return None
                
            if not self.models_trained:
                logging.error("Модели не обучены")
                return None
                
            logging.info("Проверка формы данных для полносвязной модели")
            logging.debug(f"x_test_noisy shape: {self.x_test_noisy.shape}")
            
            logging.info("Получение предсказаний полносвязной модели")
            dense_preds = self.models['dense'].predict(self.x_test_noisy)
            logging.debug(f"dense_preds shape: {dense_preds.shape}")
            
            logging.info("Проверка формы данных для сверточной модели")
            logging.debug(f"x_test_noisy_conv shape: {self.x_test_noisy_conv.shape}")
            
            logging.info("Получение предсказаний сверточной модели")
            conv_preds = self.models['conv'].predict(self.x_test_noisy_conv)
            logging.debug(f"conv_preds shape before squeeze: {conv_preds.shape}")
            conv_preds = np.squeeze(conv_preds)
            logging.debug(f"conv_preds shape after squeeze: {conv_preds.shape}")
            
            logging.info("Расчет метрик PSNR")
            dense_psnr = []
            for i in range(3):
                psnr_value = psnr(self.x_test[i], dense_preds[i])
                dense_psnr.append(psnr_value)
                logging.debug(f"Dense PSNR {i}: {psnr_value}")
                
            conv_psnr = []
            for i in range(3):
                psnr_value = psnr(self.x_test[i], conv_preds[i])
                conv_psnr.append(psnr_value)
                logging.debug(f"Conv PSNR {i}: {psnr_value}")
                
            results = {
                'noisy': self.x_test_noisy[:3],
                'dense': dense_preds[:3],
                'conv': conv_preds[:3],
                'dense_psnr': dense_psnr,
                'conv_psnr': conv_psnr
            }
            
            logging.info(f"Результаты оценки: "
                        f"Dense PSNR {np.mean(dense_psnr):.2f}, "
                        f"Conv PSNR {np.mean(conv_psnr):.2f}")
            return results
            
        except Exception as e:
            logging.exception("Ошибка при оценке моделей")
            raise
