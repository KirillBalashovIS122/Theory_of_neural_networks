import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import tensorflow_datasets as tfds
from datetime import datetime
from tqdm import tqdm
import logging
from utils import enhanced_text_preprocessing, setup_nltk_resources

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = Tokenizer(num_words=15000, oov_token="<OOV>")
        self.max_sequence_length = 200
        self.class_weights = None
        self.stop_training = False
        self.logger = logging.getLogger(self.__class__.__name__)
        setup_nltk_resources()
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных с прогресс-баром"""
        self.logger.info("Инициализация загрузки данных IMDB...")
        try:
            # Настройка параметров загрузки
            download_config = tfds.download.DownloadConfig(
                manual_dir='/tmp/imdb_data',
                extract_dir='/tmp/imdb_extracted',
                compute_stats=False
            )
            
            # Загрузка с прогресс-баром
            with tqdm(desc="Загрузка датасета", unit=" samples") as pbar:
                ds = tfds.load(
                    'imdb_reviews',
                    split='train',
                    as_supervised=True,
                    download_and_prepare_kwargs={
                        'download_config': download_config
                    }
                )
                
                texts = []
                labels = []
                for text, label in ds:
                    texts.append(text.numpy().decode('utf-8'))
                    labels.append(label.numpy())
                    pbar.update(1)
            
            df = pd.DataFrame({'text': texts, 'label': labels})
            self.logger.info(f"Загружено {len(df)} сырых записей")
            
            # Предобработка
            df['text'] = df['text'].apply(enhanced_text_preprocessing)
            df = df[df['text'].str.strip().astype(bool)]
            
            if df.empty:
                raise ValueError("Все данные были отфильтрованы")
            
            # Балансировка классов
            class_counts = df['label'].value_counts()
            self.logger.debug(f"Распределение классов:\n{class_counts}")
            
            self.class_weights = {
                0: class_counts[1] / class_counts[0] if class_counts[0] > 0 else 1.0,
                1: 1.0
            }
            
            self.logger.info(f"Осталось {len(df)} записей после очистки")
            return df
            
        except tf.errors.OpError as e:
            self.logger.error(f"Ошибка чтения данных: {e}")
            raise RuntimeError("Проверьте подключение к интернету и права доступа к файлам")
        except Exception as e:
            self.logger.error(f"Критическая ошибка загрузки: {e}", exc_info=True)
            raise

    def prepare_datasets(self, df):
        """Подготовка обучающих и тестовых данных"""
        try:
            self.logger.info("Токенизация текста...")
            self.tokenizer.fit_on_texts(df['text'])
            sequences = self.tokenizer.texts_to_sequences(df['text'])
            X = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
            y = df['label'].values
            
            self.logger.info("Балансировка классов...")
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            
            self.logger.info("Разделение на train/test...")
            return train_test_split(
                X_resampled, y_resampled, 
                test_size=0.2, 
                random_state=42,
                stratify=y_resampled
            )
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных: {e}", exc_info=True)
            raise

    def build_model(self):
        """Создание архитектуры модели"""
        try:
            model = models.Sequential([
                layers.Embedding(
                    input_dim=15000, 
                    output_dim=128,
                    input_length=self.max_sequence_length,
                    mask_zero=True
                ),
                layers.Bidirectional(layers.LSTM(
                    128, 
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    recurrent_dropout=0.2,
                    use_cudnn=False
                )),
                layers.Dropout(0.3),
                layers.Bidirectional(layers.LSTM(
                    64, 
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    use_cudnn=False
                )),
                layers.Dropout(0.2),
                layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                layers.Dense(1, activation='sigmoid')
            ])
            
            optimizer = optimizers.Adam(
                learning_rate=0.0001,
                clipnorm=1.0
            )
            
            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')
                ]
            )
            
            # Явно строим модель перед вызовом count_params
            model.build(input_shape=(None, self.max_sequence_length))
            
            self.logger.info(f"Модель успешно создана, общее количество параметров: {model.count_params()}")
            return model
            
        except Exception as e:
            self.logger.error(f"Ошибка создания модели: {e}", exc_info=True)
            raise

    def train(self, df, progress_callback=None):
        """Обучение модели с обработкой прерываний"""
        try:
            self.stop_training = False
            X_train, X_test, y_train, y_test = self.prepare_datasets(df)
            self.model = self.build_model()
            
            # Колбеки
            callbacks_list = [
                callbacks.ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True,
                    verbose=0
                ),
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=0
                ),
                self.CustomLoggingCallback(self.logger, batch_size=256),
                self.StopTrainingCallback()
            ]
            
            self.logger.info("Начало обучения...")
            history = self.model.fit(
                X_train, y_train,
                epochs=5,
                batch_size=256,
                validation_split=0.2,
                class_weight=self.class_weights,
                callbacks=callbacks_list,
                verbose=1
            )
            
            if self.stop_training:
                self.logger.warning("Обучение прервано пользователем")
                raise KeyboardInterrupt("Training interrupted by user")
                
            self.model = models.load_model('best_model.keras')
            self.logger.info("Обучение успешно завершено")
            
            return X_test, y_test
            
        except KeyboardInterrupt:
            self.logger.warning("Обучение прервано")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка обучения: {e}", exc_info=True)
            raise

    def evaluate(self, X_test, y_test):
        """Оценка качества модели"""
        try:
            self.logger.info("Оценка модели на тестовых данных...")
            y_pred = (self.model.predict(X_test) > 0.5).astype(int)
            report = classification_report(
                y_test, y_pred, 
                target_names=['Negative', 'Positive'],
                output_dict=True
            )
            
            self.logger.info(
                f"Точность: {report['accuracy']:.4f}\n"
                f"Отчет:\n{classification_report(y_test, y_pred)}"
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Ошибка оценки: {e}", exc_info=True)
            raise

    def predict_sentiment(self, text):
        """Предсказание тональности текста"""
        try:
            if not self.model:
                raise ValueError("Модель не обучена")
                
            processed_text = enhanced_text_preprocessing(text)
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded = pad_sequences(sequence, maxlen=self.max_sequence_length, padding='post')
            
            prediction = self.model.predict(padded, verbose=0)[0][0]
            self.logger.debug(
                f"Предсказание для текста: {text[:50]}...\n"
                f"Результат: {prediction:.4f}"
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Ошибка предсказания: {e}", exc_info=True)
            raise

    class CustomLoggingCallback(callbacks.Callback):
        """Кастомный колбек для логирования"""
        def __init__(self, logger, batch_size):
            super().__init__()
            self.logger = logger
            self.batch_size = batch_size
            self.start_time = None
            
        def on_train_begin(self, logs=None):
            self.start_time = datetime.now()
            self.logger.info(
                f"Начало обучения\n"
                f"Размер батча: {self.batch_size}\n"
                f"Всего эпох: {self.params['epochs']}"
            )
            
        def on_epoch_end(self, epoch, logs=None):
            current_lr = float(self.model.optimizer.learning_rate)
            epoch_time = (datetime.now() - self.start_time).total_seconds() / (epoch + 1)
            
            self.logger.info(
                f"Эпоха {epoch + 1}/{self.params['epochs']}\n"
                f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}\n"
                f"val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}\n"
                f"LR: {current_lr:.6f} - Время/эпоху: {epoch_time:.2f} сек"
            )
            
        def on_train_end(self, logs=None):
            total_time = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(
                f"Обучение завершено\n"
                f"Общее время: {total_time:.2f} сек"
            )

    class StopTrainingCallback(callbacks.Callback):
        """Колбек для обработки прерывания обучения"""
        def on_epoch_end(self, epoch, logs=None):
            if self.model.stop_training:
                self.model.stop_training = True