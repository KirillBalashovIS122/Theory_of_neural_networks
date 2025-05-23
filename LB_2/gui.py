"""
Модуль графического интерфейса приложения для подавления шума на изображениях.
Содержит реализацию главного окна приложения с элементами управления,
визуализацией результатов и системой логгирования.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.titlesize'] = 8
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import ImageDenoiser
import logging
import sys
import threading
from tkinter.scrolledtext import ScrolledText

class DenoisingAppGUI(tk.Tk):
    """
    Главное окно приложения для подавления шума на изображениях.
    
    Attributes:
        noise_factor (tk.DoubleVar): Уровень шума (0.1-0.5)
        epochs (tk.IntVar): Количество эпох обучения
        denoiser (ImageDenoiser): Обработчик данных и моделей
    """
    
    def __init__(self):
        """
        Инициализирует главное окно приложения.
        
        Настраивает:
            - Заголовок и размеры окна
            - Переменные приложения
            - UI-компоненты
            - Систему логгирования
        """
        super().__init__()
        self.title("Шумоподавитель")
        self.geometry("1000x700")
        self.configure(bg='#F0F0F0')
        self._initialize_variables()
        self._create_widgets()
        self._create_log_widget()
        self._setup_logging()

    def _initialize_variables(self):
        """Инициализирует управляющие переменные и обработчик данных."""
        self.noise_factor = tk.DoubleVar(value=0.3)
        self.epochs = tk.IntVar(value=8)
        self.denoiser = ImageDenoiser()

    def _create_widgets(self):
        """Создает основные элементы интерфейса."""
        control_frame = ttk.LabelFrame(self, text="Управление", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._create_controls(control_frame)
        self._create_result_frame()

    def _create_controls(self, parent: ttk.Frame):
        """
        Создает панель управления в указанном контейнере.
        
        Args:
            parent: Родительский контейнер для размещения элементов
            
        Элементы управления:
            - Регулятор уровня шума
            - Поле ввода эпох
            - Кнопки управления приложением
        """
        ttk.Label(parent, text="Уровень шума:").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(parent, from_=0.1, to=0.5, variable=self.noise_factor,
                command=lambda v: self._update_noise_param(float(v))).grid(row=0, column=1)
        
        ttk.Label(parent, text="Эпохи обучения:").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(parent, textvariable=self.epochs).grid(row=1, column=1)
        
        buttons = [
            ("Загрузить данные", self._load_data),
            ("Обучить модели", self._train_models),
            ("Показать результаты", self._show_results),
            ("Выход", self.destroy)
        ]
        for row, (text, command) in enumerate(buttons, start=2):
            ttk.Button(parent, text=text, command=command).grid(
                row=row, column=0, columnspan=2, pady=5)

    def _create_result_frame(self):
        """Создает контейнер для визуализации результатов обработки."""
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_log_widget(self):
        """Создает и настраивает виджет логгирования."""
        self.log_frame = ttk.LabelFrame(self, text="Лог выполнения")
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        self.log_widget = ScrolledText(self.log_frame, state='disabled', height=8)
        self.log_widget.pack(fill=tk.X, padx=5, pady=5)
        self.log_widget.configure(font=('TkFixedFont', 8))

    def _setup_logging(self):
        """
        Настраивает многоуровневую систему логгирования.
        
        Включает:
            - Вывод в текстовый виджет интерфейса
            - Консольный вывод
            - Запись в файл app.log
            - Фильтрацию сообщений matplotlib
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        class MatplotlibFilter(logging.Filter):
            def filter(self, record):
                return "findfont:" not in record.getMessage()
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s → %(message)s')
        
        widget_handler = WidgetHandler(self.log_widget)
        widget_handler.setFormatter(formatter)
        widget_handler.addFilter(MatplotlibFilter())
        self.logger.addHandler(widget_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(MatplotlibFilter())
        self.logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(MatplotlibFilter())
        self.logger.addHandler(file_handler)
        
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
        
        sys.stdout = TextRedirector(self.log_widget, 'stdout')
        sys.stderr = TextRedirector(self.log_widget, 'stderr')

    def _update_noise_param(self, value: float):
        """
        Обновляет параметр шума в модели и данных.
        
        Args:
            value: Новое значение уровня шума (0.1-0.5)
            
        Raises:
            Exception: При ошибке обновления данных
        """
        try:
            self.denoiser.noise_factor = value
            if self.denoiser.data_loaded:
                self.denoiser.update_noise()
                logging.info(f"Уровень шума обновлен до {value:.2f}")
        except Exception as e:
            logging.error(f"Ошибка обновления шума: {str(e)}", exc_info=True)

    def _load_data(self):
        """
        Загружает и предобрабатывает набор данных Fashion MNIST.
        
        Выполняет:
            - Загрузку данных через ImageDenoiser
            - Обработку исключений
            - Уведомление пользователя
        """
        try:
            logging.debug("Инициализация загрузки данных...")
            self.denoiser.load_data()
            logging.info("Данные успешно загружены и подготовлены")
            messagebox.showinfo("Успех", "Данные Fashion MNIST загружены!")
        except Exception as e:
            error_msg = f"Ошибка загрузки данных: {str(e)}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Ошибка", error_msg)

    def _train_models(self):
        """
        Запускает процесс обучения моделей в отдельном потоке.
        
        Проверяет:
            - Загрузку данных перед обучением
            - Валидность количества эпох
        """
        try:
            if not self.denoiser.data_loaded:
                warn_msg = "Необходимо сначала загрузить данные!"
                logging.warning(warn_msg)
                messagebox.showwarning("Внимание", warn_msg)
                return
                
            logging.debug(f"Подготовка к обучению на {self.epochs.get()} эпохах")
            self.denoiser.epochs = self.epochs.get()
            progress = self._create_progress_bar()
            
            threading.Thread(
                target=self._perform_training,
                args=(progress,),
                daemon=True
            ).start()
        except Exception as e:
            error_msg = f"Ошибка в процессе обучения: {str(e)}"
            logging.error(error_msg, exc_info=True)
            messagebox.showerror("Ошибка", error_msg)

    def _create_progress_bar(self) -> ttk.Progressbar:
        """
        Создает индикатор прогресса обучения.
        
        Returns:
            Объект Progressbar для отслеживания процесса
        """
        progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        progress.pack(pady=10)
        self.update()
        return progress

    def _perform_training(self, progress: ttk.Progressbar):
        """
        Выполняет обучение моделей с обновлением прогресса.
        
        Args:
            progress: Индикатор прогресса для обновления
            
        Содержит:
            - Создание архитектур моделей
            - Цикл обучения с обратным вызовом
            - Обработку завершения/ошибок
        """
        try:
            logging.debug("Создание архитектур моделей перед обучением...")
            self.denoiser.build_models()
            logging.debug("Модели успешно созданы, начало обучения...")
            
            def update_progress(epoch, logs=None):
                progress_value = (epoch + 1) * 100 / self.denoiser.epochs
                self.after(0, lambda: progress.configure(value=progress_value))
                logging.debug(f"Эпоха {epoch+1}/{self.denoiser.epochs} завершена. Loss: {logs['loss']:.4f}")
            
            logging.info("Старт обучения моделей...")
            self.denoiser.train_models(progress_callback=update_progress)
            
            self.after(0, progress.destroy)
            self.after(0, lambda: messagebox.showinfo("Успех", "Обучение завершено!"))
            logging.info("Обучение завершено успешно")
        except Exception as e:
            logging.error(f"Ошибка в процессе обучения: {str(e)}", exc_info=True)
            self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))

    def _show_results(self):
        """
        Инициирует отображение результатов обработки.
        
        Проверяет:
            - Статус обучения моделей
            - Запускает асинхронную оценку
        """
        logging.debug("Запрос на отображение результатов...")
        try:
            if not self.denoiser.models_trained:
                warn_msg = "Модели не обучены!"
                logging.warning(warn_msg)
                messagebox.showwarning("Внимание", warn_msg)
                return
                
            threading.Thread(
                target=self._async_show_results, 
                daemon=True
            ).start()
        except Exception as e:
            error_msg = f"Ошибка отображения: {str(e)}"
            logging.exception(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _async_show_results(self):
        """
        Выполняет асинхронную оценку моделей и подготовку результатов.
        
        Обрабатывает:
            - Оценку качества моделей
            - Обработку полученных результатов
            - Отображение ошибок оценки
        """
        try:
            logging.debug("Начало асинхронной оценки моделей...")
            results = self.denoiser.evaluate_models()
            
            if not results:
                logging.error("Нет данных для отображения")
                self.after(0, lambda: messagebox.showerror("Ошибка", "Нет данных для отображения"))
                return
                
            self.after(0, lambda: self._display_results(results))
        except Exception as e:
            error_msg = f"Ошибка оценки моделей: {str(e)}"
            logging.exception(error_msg)
            self.after(0, lambda: messagebox.showerror("Ошибка", error_msg))

    def _display_results(self, results: dict):
        """
        Визуализирует результаты обработки в графическом интерфейсе.
        
        Args:
            results: Словарь с результатами:
                - Зашумленные изображения
                - Обработанные изображения
                - Метрики качества
        
        Создает:
            - Матрицу изображений 3x3
            - Отображение PSNR для каждой модели
        """
        try:
            plt.close('all')
            fig = plt.figure(figsize=(10, 6))
            
            for i in range(3):
                self._add_subplot(fig, 3, 3, i+1, results['noisy'][i], "Зашумленное")
                self._add_subplot(fig, 3, 3, i+4, results['dense'][i], 
                                f"Полносвязная\nPSNR: {results['dense_psnr'][i]:.2f}")
                self._add_subplot(fig, 3, 3, i+7, results['conv'][i], 
                                f"Сверточная\nPSNR: {results['conv_psnr'][i]:.2f}")
            
            self._clear_results()
            
            canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            logging.info("Результаты успешно отображены")
        except Exception as e:
            logging.exception("Ошибка визуализации результатов")
            messagebox.showerror("Ошибка", f"Не удалось отобразить результаты: {str(e)}")

    def _clear_results(self):
        """Очищает контейнер результатов от предыдущих элементов."""
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def _add_subplot(self, fig: plt.Figure, rows: int, cols: int, index: int, 
                    image: np.ndarray, title: str):
        """
        Добавляет subplot к фигуре matplotlib.
        
        Args:
            fig: Родительская фигура
            rows: Количество строк сетки
            cols: Количество колонок сетки
            index: Позиция subplot
            image: Изображение для отображения
            title: Заголовок subplot
        """
        ax = fig.add_subplot(rows, cols, index)
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')

class WidgetHandler(logging.Handler):
    """
    Обработчик логов для интеграции с Tkinter-виджетом.
    
    Attributes:
        widget: Целевой виджет для вывода сообщений
    """
    
    def __init__(self, widget: ScrolledText):
        """
        Args:
            widget: Виджет ScrolledText для вывода логов
        """
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord):
        """
        Обрабатывает лог-запись и выводит в виджет.
        
        Args:
            record: Запись логгирования
        """
        msg = self.format(record)
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, msg + '\n')
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

class TextRedirector:
    """
    Перенаправитель вывода stdout/stderr в Tkinter-виджет.
    
    Attributes:
        widget: Целевой виджет
        tag: Тег форматирования
    """
    
    def __init__(self, widget: ScrolledText, tag: str):
        """
        Args:
            widget: Виджет для перенаправления
            tag: Идентификатор формата вывода
        """
        self.widget = widget
        self.tag = tag

    def write(self, text: str):
        """
        Записывает текст в виджет.
        
        Args:
            text: Выводимый текст
        """
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, text, self.tag)
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

    def flush(self):
        """Заглушка для совместимости с интерфейсом потока."""
        pass
