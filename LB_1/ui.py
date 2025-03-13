from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QPushButton, QLabel, 
                            QProgressBar, QWidget, QSlider, QSpinBox, QHBoxLayout,
                            QMessageBox, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from model import TrainingThread
from config import config
from utils import generate_music_sequence, save_midi, play_midi

class MusicGeneratorUI(QMainWindow):
    """
    Главный класс графического интерфейса приложения.

    Атрибуты:
        model (keras.Model): Обученная модель для генерации музыки.
        playback_process (subprocess.Popen): Процесс воспроизведения музыки.
    """
    def __init__(self):
        """Инициализация графического интерфейса."""
        super().__init__()
        self.model = None
        self.playback_process = None
        self._init_ui()
        self._setup_connections()

    def _init_ui(self):
        """Инициализация элементов интерфейса."""
        self.setWindowTitle("Нейромузыкальный генератор")
        self.setGeometry(100, 100, 800, 600)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        self._create_widgets()
        self._setup_layout(layout)

    def _create_widgets(self):
        """Создание виджетов интерфейса."""
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Готово")
        self.train_btn = QPushButton("Обучить модель")
        self.generate_btn = QPushButton("Сгенерировать музыку")
        self.play_btn = QPushButton("▶ Воспроизвести")
        self.slider = QSlider(Qt.Horizontal)
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.file_spin = QSpinBox()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Простая RNN", "LSTM", "GRU"])

    def _setup_layout(self, layout):
        """Настройка расположения виджетов."""
        self.file_spin.setRange(1, 1000)
        self.file_spin.setValue(10)
        
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Файлов для обучения:"))
        file_layout.addWidget(self.file_spin)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Архитектура модели:"))
        model_layout.addWidget(self.model_selector)
        
        layout.addLayout(file_layout)
        layout.addLayout(model_layout)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.train_btn)
        layout.addWidget(self.generate_btn)
        layout.addWidget(self.canvas)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.slider)

    def _setup_connections(self):
        """Настройка сигналов и слотов."""
        self.train_btn.clicked.connect(self.start_training)
        self.generate_btn.clicked.connect(self.generate)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.slider.sliderMoved.connect(self.seek)

    def start_training(self):
        """Запуск процесса обучения модели."""
        self.train_btn.setEnabled(False)
        model_type = ["simple", "lstm", "gru"][self.model_selector.currentIndex()]
        self.thread = TrainingThread(config.DATA_DIR, self.file_spin.value(), model_type)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.training_finished.connect(self.training_complete)
        self.thread.error_occurred.connect(self.show_error)
        self.thread.start()

    def generate(self):
        """Генерация новой музыкальной последовательности."""
        if not self.model:
            self.show_error("Модель не обучена!")
            return
        
        sequence = generate_music_sequence(self.model, config.SEQ_LENGTH, config.GENERATION_LENGTH)
        save_midi(sequence, config.NOTE_DURATION)
        self.plot_sequence(sequence)
        self.slider.setRange(0, int(config.GENERATION_LENGTH * config.NOTE_DURATION))
        self.play_btn.setEnabled(True)

    def toggle_playback(self):
        """Управление воспроизведением/паузой."""
        if self.playback_process:
            self.stop_playback()
        else:
            self.start_playback()

    def update_progress(self, progress_percent, loss, val_loss):
        """
        Обновление прогресса обучения.

        Аргументы:
            progress_percent (int): Процент выполнения обучения.
            loss (float): Значение функции потерь.
            val_loss (float): Значение валидационной функции потерь.
        """
        self.progress_bar.setValue(progress_percent)
        self.status_label.setText(f"Обучение: {progress_percent}% | Потери: {loss:.4f} | Валидационные потери: {val_loss:.4f}")

    def plot_sequence(self, sequence):
        """
        Визуализация сгенерированной последовательности.

        Аргументы:
            sequence (list): Список нот для отображения.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(sequence)
        ax.set_title("Сгенерированная последовательность нот")
        ax.set_xlabel("Индекс ноты")
        ax.set_ylabel("Высота ноты")
        self.canvas.draw()

    def training_complete(self):
        """Обработчик завершения обучения."""
        self.model = self.thread.model
        self.train_btn.setEnabled(True)
        self.status_label.setText("Обучение завершено")

    def start_playback(self):
        """Начало воспроизведения музыки."""
        self.playback_process = play_midi('temp.mid')
        self.play_btn.setText("⏸ Пауза")

    def stop_playback(self):
        """Остановка воспроизведения."""
        if self.playback_process:
            self.playback_process.terminate()
            self.playback_process = None
        self.play_btn.setText("▶ Воспроизвести")

    def seek(self, position):
        """Перемотка музыки."""
        pass

    def show_error(self, message):
        """
        Отображение сообщения об ошибке.

        Аргументы:
            message (str): Текст сообщения об ошибке.
        """
        QMessageBox.critical(self, "Ошибка", message)
        self.status_label.setText(f"Ошибка: {message}")
        self.train_btn.setEnabled(True)
