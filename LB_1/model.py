import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import glob
import pretty_midi
import time
from PyQt5.QtCore import QThread, pyqtSignal
from config import config

class TrainingThread(QThread):
    """
    Поток для обучения модели в фоновом режиме.
    
    Сигналы:
        progress_updated (int, float, float): Прогресс обучения, потери, валидационные потери.
        training_finished: Сигнал завершения обучения.
        error_occurred (str): Сигнал об ошибке.
    """
    progress_updated = pyqtSignal(int, float, float)
    training_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, data_dir, max_files, model_type):
        """
        Инициализация потока обучения.

        Аргументы:
            data_dir (Path): Путь к директории с MIDI-файлами.
            max_files (int): Максимальное количество файлов для обучения.
            model_type (str): Тип модели (simple, lstm, gru).
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_files = max_files
        self.model_type = model_type
        self.start_time = None

    def run(self):
        """Основной метод выполнения потока обучения."""
        try:
            file_list = glob.glob(str(self.data_dir / '**/*.mid*'), recursive=True)
            if not file_list:
                raise FileNotFoundError("MIDI файлы не найдены")
            
            file_list = file_list[:self.max_files] if self.max_files > 0 else file_list
            all_notes = self._process_files(file_list)
            X, y = self._create_sequences(all_notes)
            self.model = self._build_model()
            self._train_model(X, y)
            self.training_finished.emit()
            
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _process_files(self, file_list):
        """
        Обработка MIDI-файлов и извлечение нот.

        Аргументы:
            file_list (list): Список путей к MIDI-файлам.

        Возвращает:
            list: Список всех нот из обработанных файлов.
        """
        all_notes = []
        for file in file_list:
            pm = pretty_midi.PrettyMIDI(file)
            all_notes.extend(note.pitch for note in pm.instruments[0].notes)
        return all_notes

    def _create_sequences(self, notes):
        """
        Создание обучающих последовательностей.

        Аргументы:
            notes (list): Список нот.

        Возвращает:
            tuple: (X, y) - входные последовательности и целевые значения.
        """
        sequences = []
        targets = []
        for i in range(len(notes) - config.SEQ_LENGTH):
            sequences.append(notes[i:i + config.SEQ_LENGTH])
            targets.append(notes[i + config.SEQ_LENGTH])
        return np.array(sequences), np.array(targets)

    def _build_model(self):
        """
        Построение архитектуры нейронной сети.

        Возвращает:
            Sequential: Собранная модель Keras.
        """
        model_params = config.MODEL_PARAMS[self.model_type]
        
        if self.model_type == "simple":
            model = Sequential([
                layers.Embedding(128, 16, input_length=config.SEQ_LENGTH),
                layers.SimpleRNN(64),
                layers.Dense(128, activation='softmax')
            ])
        elif self.model_type == "lstm":
            model = Sequential([
                layers.Embedding(128, 16, input_length=config.SEQ_LENGTH),
                layers.LSTM(64, return_sequences=True),
                layers.LSTM(32),
                layers.Dense(128, activation='softmax')
            ])
        elif self.model_type == "gru":
            model = Sequential([
                layers.Embedding(128, 16, input_length=config.SEQ_LENGTH),
                layers.GRU(128),
                layers.Dense(128, activation='softmax')
            ])
        
        model.compile(loss='sparse_categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model

    def _train_model(self, X, y):
        """
        Обучение модели.

        Аргументы:
            X (numpy.ndarray): Входные данные.
            y (numpy.ndarray): Целевые значения.
        """
        params = config.MODEL_PARAMS[self.model_type]
        self.start_time = time.time()
        self.model.fit(
            X, y,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_split=0.2,
            callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=self._log_progress)]
        )

    def _log_progress(self, epoch, logs):
        """
        Логирование прогресса обучения.

        Аргументы:
            epoch (int): Номер текущей эпохи.
            logs (dict): Метрики обучения.
        """
        elapsed = time.time() - self.start_time
        epoch_time = elapsed / (epoch + 1)
        remaining = epoch_time * (config.MODEL_PARAMS[self.model_type]["epochs"] - epoch - 1)
        
        progress_percent = int((epoch + 1) / config.MODEL_PARAMS[self.model_type]["epochs"] * 100)
        
        log = f"Эпоха {epoch+1}/{config.MODEL_PARAMS[self.model_type]['epochs']} | Потери: {logs['loss']:.4f} | Валидационные потери: {logs['val_loss']:.4f}"
        print(log)
        
        self.progress_updated.emit(progress_percent, logs['loss'], logs['val_loss'])
