"""
Модуль обработки данных и управления моделями.
Содержит логику работы с данными и обучения моделей.
"""
import numpy as np
import matplotlib as mpl
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
    """Класс для управления процессом шумоподавления.
    
    Attributes:
        noise_factor (float): Уровень шума (0.1–0.5).
        epochs (int): Количество эпох обучения.
        models (dict): Словарь с моделями ('dense' и 'conv').
        data_loaded (bool): Флаг загрузки данных.
    """
    def __init__(self, noise_factor=0.3, epochs=8):
        self.noise_factor = noise_factor
        self.epochs = epochs
        self.models = {'dense': None, 'conv': None}
        self.models_trained = False
        self.data_loaded = False
        logging.debug(f"ImageDenoiser инициализирован с noise_factor={noise_factor}, epochs={epochs}")

    def load_data(self):
        """Загружает данные Fashion MNIST и нормализует их."""
        logging.info("Загрузка данных Fashion MNIST...")
        (self.x_train, _), (self.x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
        
        logging.debug("Нормализация данных...")
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        
        self._update_noisy_data()
        self.data_loaded = True
        logging.info(f"Данные подготовлены (шум: {self.noise_factor})")

    def _update_noisy_data(self):
        """Генерирует зашумленные версии данных."""
        logging.debug("Генерация зашумленных данных...")
        self.x_train_noisy = self._add_noise(self.x_train)
        self.x_test_noisy = self._add_noise(self.x_test)
        self.x_train_conv = np.expand_dims(self.x_train, -1)
        self.x_test_conv = np.expand_dims(self.x_test, -1)
        self.x_train_noisy_conv = np.expand_dims(self.x_train_noisy, -1)
        self.x_test_noisy_conv = np.expand_dims(self.x_test_noisy, -1)
        logging.debug(f"Зашумленные данные сгенерированы: train={self.x_train_noisy.shape}, test={self.x_test_noisy.shape}")

    def update_noise(self):
        """Обновляет зашумленные данные при изменении уровня шума."""
        logging.info("Обновление шумовых параметров...")
        self._update_noisy_data()

    def _add_noise(self, data):
        """Добавляет гауссов шум к данным.
        
        Args:
            data (np.array): Исходные данные (форма (N, 28, 28)).
        
        Returns:
            np.array: Зашумленные данные.
        """
        logging.debug(f"Добавление шума к данным формы {data.shape}")
        noisy = data + self.noise_factor * np.random.normal(size=data.shape)
        return np.clip(noisy, 0., 1.)

    def build_models(self):
        """Инициализирует модели автоэнкодеров."""
        logging.info("Инициализация новых моделей...")
        self.models['dense'] = DenseAutoencoder()
        self.models['conv'] = ConvAutoencoder()
        logging.debug("Архитектуры моделей созданы: DenseAutoencoder и ConvAutoencoder")

    def train_models(self, progress_callback=None):
        """Обучает модели автоэнкодеров.
        
        Args:
            progress_callback (function): Функция для отслеживания прогресса.
        """
        logging.info("Старт обучения моделей...")
        self._train_single_model('dense', progress_callback)
        self._train_single_model('conv', progress_callback)
        self.models_trained = True
        logging.info("Все модели успешно обучены")

    def _train_single_model(self, model_type, progress_callback=None):
        """Обучает одну модель.
        
        Args:
            model_type (str): Тип модели ('dense' или 'conv').
            progress_callback (function): Функция для отслеживания прогресса.
        
        Raises:
            Exception: Если возникает ошибка при обучении.
        """
        try:
            logging.debug(f"Обучение {model_type} модели...")
            model = self.models[model_type]
            logging.debug(f"Компиляция {model_type} модели с оптимизатором 'adam' и MSE")
            model.compile(optimizer='adam', loss=losses.MeanSquaredError())
            
            train_data = self.x_train_noisy if model_type == 'dense' else self.x_train_noisy_conv
            val_data = self.x_test_noisy if model_type == 'dense' else self.x_test_noisy_conv
            target_data = self.x_train if model_type == 'dense' else self.x_train_conv
            val_target_data = self.x_test if model_type == 'dense' else self.x_test_conv
            
            logging.debug(f"Запуск обучения {model_type} модели: "
                         f"train_data={train_data.shape}, target_data={target_data.shape}, "
                         f"epochs={self.epochs}, batch_size=512")
            
            model.fit(
                train_data,
                target_data,
                epochs=self.epochs,
                batch_size=512,
                verbose=0,
                validation_data=(val_data, val_target_data),
                callbacks=[LambdaCallback(on_epoch_end=progress_callback)] 
                          if progress_callback else None
            )
            logging.debug(f"{model_type} модель обучена")
        except Exception as e:
            logging.error(f"Ошибка обучения {model_type} модели: {str(e)}", exc_info=True)
            raise

    def evaluate_models(self):
        """Оценивает качество моделей с помощью метрики PSNR.
        
        Returns:
            dict: Результаты оценки (зашумленные данные, предсказания, PSNR).
        
        Raises:
            ValueError: Если данные не загружены или модели не обучены.
        """
        logging.info("Оценка качества моделей...")
        try:
            if not self.data_loaded:
                raise ValueError("Данные не загружены")
                
            if not self.models_trained:
                raise ValueError("Модели не обучены")
                
            logging.debug("Генерация предсказаний для DenseAutoencoder...")
            dense_preds = self.models['dense'].predict(self.x_test_noisy, batch_size=512, verbose=0)
            logging.debug("Генерация предсказаний для ConvAutoencoder...")
            conv_preds = np.squeeze(self.models['conv'].predict(self.x_test_noisy_conv, batch_size=512, verbose=0))
            
            if dense_preds.shape[0] < 3 or conv_preds.shape[0] < 3:
                raise ValueError("Недостаточно данных для отображения")
            
            logging.debug("Вычисление PSNR для первых 3 изображений...")
            dense_psnr = [psnr(self.x_test[i], dense_preds[i]) for i in range(3)]
            conv_psnr = [psnr(self.x_test[i], conv_preds[i]) for i in range(3)]
            
            results = {
                'noisy': self.x_test_noisy[:3],
                'dense': dense_preds[:3],
                'conv': conv_preds[:3],
                'dense_psnr': dense_psnr,
                'conv_psnr': conv_psnr
            }
            
            logging.info(f"Результаты оценки - Dense: {np.mean(dense_psnr):.2f} dB, Conv: {np.mean(conv_psnr):.2f} dB")
            return results
            
        except Exception as e:
            logging.exception("Ошибка оценки моделей")
            raise
