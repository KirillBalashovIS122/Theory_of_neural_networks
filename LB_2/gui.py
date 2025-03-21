"""
Модуль графического интерфейса приложения.
Содержит классы для управления взаимодействием с пользователем.
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from processing import ImageDenoiser
import logging
import sys
from tkinter.scrolledtext import ScrolledText

class DenoisingAppGUI(tk.Tk):
    """Главное окно приложения с элементами управления и визуализацией"""
    def __init__(self):
        """Инициализация графического интерфейса"""
        super().__init__()
        self.title("Шумоподавитель")
        self.geometry("1000x700")
        self.configure(bg='#F0F0F0')
        
        self._initialize_variables()
        self._create_widgets()
        self._create_log_widget()
        self._setup_logging()

    def _initialize_variables(self):
        """Инициализация переменных управления"""
        self.noise_factor = tk.DoubleVar(value=0.3)
        self.epochs = tk.IntVar(value=8)
        self.denoiser = ImageDenoiser()

    def _create_widgets(self):
        """Создание элементов интерфейса"""
        control_frame = ttk.LabelFrame(self, text="Управление", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        self._create_controls(control_frame)
        self._create_result_frame()

    def _create_controls(self, parent):
        """Создание панели управления"""
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
        """Создание области для отображения результатов"""
        self.result_frame = ttk.Frame(self)
        self.result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _create_log_widget(self):
        """Создание виджета для отображения логов"""
        self.log_frame = ttk.LabelFrame(self, text="Лог выполнения")
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        self.log_widget = ScrolledText(self.log_frame, state='disabled', height=8)
        self.log_widget.pack(fill=tk.X, padx=5, pady=5)
        self.log_widget.configure(font=('TkFixedFont', 8))

    def _setup_logging(self):
        """Настройка системы логирования"""
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s | %(levelname)s → %(message)s')
        
        widget_handler = WidgetHandler(self.log_widget)
        widget_handler.setFormatter(formatter)
        self.logger.addHandler(widget_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler('app.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        sys.stdout = TextRedirector(self.log_widget, 'stdout')
        sys.stderr = TextRedirector(self.log_widget, 'stderr')

    def _update_noise_param(self, value):
        """Обновление параметра уровня шума"""
        try:
            self.denoiser.noise_factor = value
            if self.denoiser.data_loaded:
                self.denoiser.update_noise()
                logging.info(f"Уровень шума обновлен до {value}")
        except Exception as e:
            logging.error(f"Ошибка обновления шума: {str(e)}")

    def _load_data(self):
        """Загрузка и подготовка данных"""
        try:
            logging.info("Начата загрузка данных Fashion MNIST...")
            self.denoiser.load_data()
            logging.info("Данные успешно загружены и подготовлены")
            messagebox.showinfo("Успех", "Данные Fashion MNIST загружены!")
        except Exception as e:
            error_msg = f"Ошибка загрузки данных: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _train_models(self):
        """Запуск процесса обучения моделей"""
        try:
            if not self.denoiser.data_loaded:
                warn_msg = "Необходимо сначала загрузить данные!"
                logging.warning(warn_msg)
                messagebox.showwarning("Внимание", warn_msg)
                return
                
            logging.info(f"Начато обучение моделей на {self.epochs.get()} эпохах")
            self.denoiser.epochs = self.epochs.get()
            
            progress = self._create_progress_bar()
            self._perform_training(progress)
            progress.destroy()
            
            logging.info("Обучение моделей успешно завершено")
            messagebox.showinfo("Успех", "Обучение завершено!")
        except Exception as e:
            error_msg = f"Ошибка в процессе обучения: {str(e)}"
            logging.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _create_progress_bar(self):
        """Создание индикатора выполнения"""
        progress = ttk.Progressbar(self, orient="horizontal", length=200, mode="determinate")
        progress.pack(pady=10)
        self.update()
        return progress

    def _perform_training(self, progress):
        """Выполнение обучения с обновлением прогресса"""
        logging.info("Создание архитектур моделей...")
        self.denoiser.build_models()
        
        def update_progress(epoch, logs=None):
            progress_value = (epoch + 1) * 100 / self.denoiser.epochs
            progress['value'] = progress_value
            self.update()
            logging.info(f"Эпоха {epoch+1}/{self.denoiser.epochs} завершена "
                        f"(прогресс: {progress_value:.1f}%)")
        
        logging.info("Запуск процесса обучения моделей...")
        self.denoiser.train_models(progress_callback=update_progress)

    def _show_results(self):
        """Отображение результатов работы моделей"""
        logging.info("Кнопка 'Показать результаты' нажата")
        try:
            if not self.denoiser.models_trained:
                warn_msg = "Модели не обучены!"
                logging.warning(warn_msg)
                messagebox.showwarning("Внимание", warn_msg)
                return
                
            logging.info("Запуск оценки моделей")
            results = self.denoiser.evaluate_models()
            
            if not results:
                logging.error("Оценка моделей вернула пустые результаты")
                messagebox.showerror("Ошибка", "Нет данных для отображения")
                return
                
            logging.info(f"Получены результаты: {len(results['noisy'])} примеров")
            self._display_results(results)
            logging.info("Результаты успешно отображены")
        except Exception as e:
            error_msg = f"Ошибка отображения: {str(e)}"
            logging.exception(error_msg)
            messagebox.showerror("Ошибка", error_msg)

    def _display_results(self, results):
        logging.info("Начато отображение результатов")
        try:
            logging.debug(f"noisy shape: {results['noisy'].shape}")
            logging.debug(f"dense shape: {results['dense'].shape}")
            logging.debug(f"conv shape: {results['conv'].shape}")
            
            if results['noisy'].shape[0] < 3:
                raise ValueError("Недостаточно тестовых примеров")
                
            self._clear_results()
            fig = plt.figure(figsize=(10, 6))
            self._create_result_subplots(fig, results)
            self._embed_plot(fig)
            logging.info("Результаты успешно отображены")
        
        except Exception as e:
            logging.exception("Ошибка отображения результатов")
            messagebox.showerror("Ошибка", f"Не удалось отобразить результаты: {str(e)}")

    def _clear_results(self):
        """Очистка области результатов"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()

    def _create_result_subplots(self, fig, results):
        """Создание подграфиков с результатами"""
        for i in range(3):
            self._add_subplot(fig, 3, 3, i+1, results['noisy'][i], "Зашумленное")
            self._add_subplot(fig, 3, 3, i+4, results['dense'][i], 
                            f"Полносвязная\nPSNR: {results['dense_psnr'][i]:.2f}")
            self._add_subplot(fig, 3, 3, i+7, results['conv'][i], 
                            f"Сверточная\nPSNR: {results['conv_psnr'][i]:.2f}")

    def _add_subplot(self, fig, rows, cols, index, image, title):
        """Добавление подграфика с изображением"""
        ax = fig.add_subplot(rows, cols, index)
        ax.imshow(image, cmap='gray')
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    def _embed_plot(self, fig):
        """Встраивание графика в интерфейс"""
        canvas = FigureCanvasTkAgg(fig, master=self.result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

class WidgetHandler(logging.Handler):
    """Обработчик логов для Tkinter"""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, msg + '\n')
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

class TextRedirector:
    """Перенаправитель вывода в виджет"""
    def __init__(self, widget, tag):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, text, self.tag)
        self.widget.configure(state='disabled')
        self.widget.yview(tk.END)

    def flush(self):
        pass
