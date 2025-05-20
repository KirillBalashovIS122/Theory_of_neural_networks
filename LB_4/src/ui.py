import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, StringVar
from threading import Thread
from tqdm import tqdm
import logging
import queue

class UIHandler:
    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.logger = logging.getLogger(self.__class__.__name__)
        self.progress_queue = queue.Queue()
        self.test_samples = {
            "Негативный (пример)": "This movie was completely awful and a waste of time!",
            "Позитивный": "An amazing film with brilliant performances!",
            "Саркастический": "Wow, what a 'fantastic' way to bore the audience!",
            "Нейтральный": "The movie is about a group of friends on an adventure."
        }
        
        self.create_widgets()
        self.setup_status_bar()
        self.check_queue()
        
    def create_widgets(self):
        """Создание элементов интерфейса"""
        self.create_control_frame()
        self.create_classify_frame()
        
    def create_control_frame(self):
        """Панель управления"""
        control_frame = ttk.LabelFrame(self.root, text="Управление моделью", padding=10)
        control_frame.pack(padx=10, pady=5, fill=tk.X)
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.load_btn = ttk.Button(
            btn_frame, 
            text="Загрузить данные", 
            command=self.load_data
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.train_btn = ttk.Button(
            btn_frame, 
            text="Обучить модель", 
            command=self.start_training,
            state=tk.DISABLED
        )
        self.train_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(
            btn_frame, 
            text="Сохранить модель", 
            command=self.save_model,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(
            control_frame, 
            orient=tk.HORIZONTAL, 
            mode='determinate',
            maximum=100
        )
        self.progress.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(
            control_frame,
            text="Готов к работе",
            anchor=tk.W
        )
        self.progress_label.pack(fill=tk.X)
        
    def create_classify_frame(self):
        """Фрейм классификации"""
        classify_frame = ttk.LabelFrame(self.root, text="Анализ отзыва", padding=10)
        classify_frame.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Выпадающее меню для тестовых примеров
        sample_frame = ttk.Frame(classify_frame)
        sample_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sample_frame, text="Выберите пример:").pack(side=tk.LEFT, padx=5)
        self.sample_combobox = ttk.Combobox(
            sample_frame,
            values=list(self.test_samples.keys()),
            state="readonly",
            width=30
        )
        self.sample_combobox.pack(side=tk.LEFT, padx=5)
        self.sample_combobox.bind("<<ComboboxSelected>>", self.insert_sample_text)
        
        # Поле ввода текста
        self.text_input = scrolledtext.ScrolledText(
            classify_frame, 
            height=10, 
            wrap=tk.WORD,
            font=('Arial', 10)
        )
        self.text_input.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель управления классификацией
        btn_frame = ttk.Frame(classify_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            btn_frame, 
            text="Проверить отзыв", 
            command=self.predict
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame, 
            text="Очистить поле", 
            command=self.clear_input
        ).pack(side=tk.LEFT, padx=5)
        
        # Отображение результата
        self.result_var = StringVar(value="Результат: ")
        self.result_label = ttk.Label(
            classify_frame, 
            textvariable=self.result_var,
            font=('Arial', 10, 'bold'),
            wraplength=500,
            padding=5
        )
        self.result_label.pack(fill=tk.X, padx=5, pady=5)
        
    def insert_sample_text(self, event):
        """Вставка выбранного тестового примера в поле ввода"""
        selected_sample = self.sample_combobox.get()
        if selected_sample:
            self.text_input.delete("1.0", tk.END)
            self.text_input.insert(tk.END, self.test_samples[selected_sample])
        
    def setup_status_bar(self):
        """Строка состояния"""
        self.status_var = StringVar(value="Готов к работе")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN,
            padding=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def load_data(self):
        """Загрузка данных в отдельном потоке"""
        def data_loading_thread():
            try:
                self.update_controls_state(tk.DISABLED)
                self.status_var.set("Загрузка данных...")
                self.progress_label.config(text="Загрузка IMDB датасета...")
                
                df = self.app.model_handler.load_and_prepare_data()
                self.df = df  # Сохранение DataFrame в self.df
                
                self.progress_queue.put({
                    'type': 'data_loaded',
                    'message': f"Загружено {len(df)} записей",
                    'df_size': len(df)
                })
                
            except Exception as e:
                self.progress_queue.put({
                    'type': 'error',
                    'message': str(e)
                })
        
        Thread(target=data_loading_thread, daemon=True).start()
        
    def start_training(self):
        """Запуск обучения в отдельном потоке"""
        def training_thread():
            try:
                self.update_controls_state(tk.DISABLED)
                self.status_var.set("Подготовка к обучению...")
                self.progress_label.config(text="Подготовка данных...")
                
                X_test, y_test = self.app.model_handler.train(
                    self.df,
                    progress_callback=self.update_progress
                )
                
                report = self.app.model_handler.evaluate(X_test, y_test)
                
                self.progress_queue.put({
                    'type': 'training_complete',
                    'report': report
                })
                
            except Exception as e:
                self.progress_queue.put({
                    'type': 'error',
                    'message': str(e)
                })
        
        if hasattr(self, 'df'):
            Thread(target=training_thread, daemon=True).start()
        else:
            messagebox.showwarning("Ошибка", "Сначала загрузите данные!")
            
    def update_progress(self, progress_info):
        """Обновление прогресса через очередь"""
        self.progress_queue.put({
            'type': 'progress',
            'info': progress_info
        })
        
    def check_queue(self):
        """Проверка очереди сообщений"""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                
                if message['type'] == 'data_loaded':
                    self.status_var.set(f"Загружено {message['df_size']} записей")
                    self.progress_label.config(text="Данные готовы к обучению")
                    self.train_btn.config(state=tk.NORMAL)
                    messagebox.showinfo("Успех", message['message'])
                    
                elif message['type'] == 'progress':
                    info = message['info']
                    self.progress['value'] = (info['epoch'] / info['total_epochs']) * 100
                    self.status_var.set(
                        f"Эпоха {info['epoch']}/{info['total_epochs']} | "
                        f"Loss: {info['logs']['loss']:.4f} | "
                        f"Accuracy: {info['logs']['accuracy']:.4f}"
                    )
                    
                elif message['type'] == 'training_complete':
                    self.status_var.set("Обучение завершено")
                    self.progress['value'] = 100
                    self.progress_label.config(text="Модель готова к использованию")
                    self.save_btn.config(state=tk.NORMAL)
                    self.show_report(message['report'])
                    
                elif message['type'] == 'error':
                    self.status_var.set("Ошибка")
                    self.progress_label.config(text="Произошла ошибка")
                    messagebox.showerror("Ошибка", message['message'])
                    
                self.update_controls_state(tk.NORMAL)
                
        except queue.Empty:
            pass
            
        self.root.after(100, self.check_queue)
        
    def update_controls_state(self, state):
        """Обновление состояния элементов управления"""
        self.load_btn.config(state=state)
        self.train_btn.config(state=state if hasattr(self, 'df') else tk.DISABLED)
        self.save_btn.config(state=state if self.app.model_handler.model else tk.DISABLED)
        
    def predict(self):
        """Обработка предсказания"""
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Ошибка", "Введите текст для анализа!")
            return
            
        try:
            prediction = self.app.model_handler.predict_sentiment(text)
            result = "Позитивный" if prediction > 0.5 else "Негативный"
            confidence = prediction if result == "Позитивный" else 1 - prediction
            
            self.result_var.set(
                f"Результат: {result}\n"
                f"Уверенность: {confidence*100:.1f}%\n"
                f"Оценка модели: {prediction:.4f}"
            )
            self.result_label.config(
                foreground="green" if result == "Позитивный" else "red"
            )
            
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            
    def clear_input(self):
        """Очистка поля ввода"""
        self.text_input.delete("1.0", tk.END)
        
    def save_model(self):
        """Сохранение модели"""
        try:
            path = filedialog.asksaveasfilename(
                defaultextension=".keras",
                filetypes=[("Keras model", "*.keras"), ("All files", "*.*")],
                title="Сохранить модель как"
            )
            
            if path:
                self.app.model_handler.model.save(path)
                messagebox.showinfo("Сохранено", f"Модель сохранена в:\n{path}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить модель:\n{e}")
            
    def show_report(self, report):
        """Отображение отчета о классификации"""
        report_window = tk.Toplevel(self.root)
        report_window.title("Результаты классификации")
        
        text_area = scrolledtext.ScrolledText(
            report_window,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Courier', 10)
        )
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        report_text = (
            f"{'Метрика':<15}{'Точность':<10}{'Полнота':<10}{'F1-score':<10}\n"
            f"{'-'*45}\n"
            f"{'Negative':<15}{report['Negative']['precision']:.3f}{'':<10}{report['Negative']['recall']:.3f}{'':<10}{report['Negative']['f1-score']:.3f}\n"
            f"{'Positive':<15}{report['Positive']['precision']:.3f}{'':<10}{report['Positive']['recall']:.3f}{'':<10}{report['Positive']['f1-score']:.3f}\n"
            f"{'-'*45}\n"
            f"{'Accuracy':<15}{'':<10}{'':<10}{report['accuracy']:.3f}\n"
        )
        
        text_area.insert(tk.END, report_text)
        text_area.config(state=tk.DISABLED)