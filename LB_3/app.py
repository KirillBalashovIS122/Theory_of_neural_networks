import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from model import SegmentationModel

class SegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Сегментация изображений")
        self.master.geometry("1200x800")
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.model_handler = SegmentationModel(
            base_dir=self.BASE_DIR,
            image_size=(256, 256),
            classes={
                'cloudy': 0,
                'desert': 1,
                'green_area': 2,
                'water': 3
            }
        )
        self.current_image = None
        self.current_image_path = ""
        self.model_loaded = False
        self.setup_ui()
        self.check_model()

    def setup_ui(self):
        main_frame = tk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right_frame = tk.Frame(main_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=0)
        left_frame.grid_columnconfigure(0, weight=1)
        img_display_frame = tk.Frame(left_frame)
        img_display_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        self.original_img = tk.Label(img_display_frame, bg='white')
        self.original_img.pack(pady=10, fill=tk.X)
        self.result_img = tk.Label(img_display_frame, bg='white')
        self.result_img.pack(pady=10, fill=tk.X)
        btn_frame = tk.Frame(left_frame)
        btn_frame.grid(row=1, column=0, sticky='ew', pady=10)
        tk.Button(btn_frame, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Обучить модель", command=self.start_training).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Сегментировать", command=self.segment_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Импорт модели", command=self.import_model).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Выход", command=self.master.quit).pack(side=tk.LEFT, padx=5)
        info_frame = tk.LabelFrame(right_frame, text="Информация", padx=10, pady=10)
        info_frame.pack(fill=tk.BOTH, padx=10, pady=10)
        tk.Label(info_frame, text="Текущая модель:", anchor='w').pack(fill=tk.X)
        self.model_info = tk.Label(info_frame, text="Модель не загружена", anchor='w', fg="blue")
        self.model_info.pack(fill=tk.X)
        color_frame = tk.LabelFrame(info_frame, text="Цвета сегментации", padx=5, pady=5)
        color_frame.pack(fill=tk.X, pady=10)
        colors = [
            ("Красный", "Облака (cloudy)"),
            ("Желтый", "Пустыня (desert)"),
            ("Зеленый", "Зелень (green_area)"),
            ("Синий", "Вода (water)")
        ]
        for color, label in colors:
            tk.Label(color_frame, text=f"{color}: {label}", anchor='w').pack(fill=tk.X)
        tk.Label(info_frame, text="Текущее изображение:", anchor='w').pack(fill=tk.X)
        self.image_info = tk.Label(info_frame, text="Изображение не загружено", anchor='w', fg="blue")
        self.image_info.pack(fill=tk.X)

    def check_model(self):
        if self.model_handler.load_saved_model():
            self.model_loaded = True
            self.update_model_info("models/best_model.keras")
        else:
            self.show_model_dialog()

    def update_model_info(self, text):
        self.model_info.config(text=text)
        self.model_loaded = True

    def update_image_info(self, text):
        filename = os.path.basename(text)
        self.image_info.config(text=filename)
        self.current_image_path = text

    def show_model_dialog(self):
        if messagebox.askyesno(
            "Модель не найдена", 
            "Обученная модель отсутствует. Обучить новую модель?",
            parent=self.master
        ):
            self.start_training()
        else:
            self.master.quit()

    def load_image(self):
        path = filedialog.askopenfilename(
            initialdir=self.BASE_DIR,
            filetypes=[("Изображения", "*.png *.jpg *.jpeg")],
            title="Выберите изображение"
        )
        if path:
            try:
                self.current_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                self.display_image(self.current_image, self.original_img)
                self.update_image_info(path)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def segment_image(self):
        if not self.model_loaded:
            messagebox.showerror("Ошибка", "Сначала загрузите или обучите модель!")
            return
        if self.current_image is None:
            messagebox.showerror("Ошибка", "Сначала загрузите изображение!")
            return
        try:
            img = cv2.resize(self.current_image, (256, 256)) / 255.0
            img_flipped = np.fliplr(img)
            prediction_original = self.model_handler.model.predict(np.expand_dims(img, axis=0))[0]
            prediction_flipped = self.model_handler.model.predict(np.expand_dims(img_flipped, axis=0))[0]
            prediction_flipped_back = np.fliplr(prediction_flipped)
            prediction_avg = (prediction_original + prediction_flipped_back) / 2
            mask = np.argmax(prediction_avg, axis=-1)
            colored_mask = self.colorize_mask(mask)
            self.display_image(colored_mask, self.result_img)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сегментации: {str(e)}")

    def colorize_mask(self, mask):
        color_map = {
            0: [255, 0, 0],
            1: [255, 255, 0],
            2: [0, 255, 0],
            3: [0, 0, 255]
        }
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            colored[mask == class_id] = color
        return colored

    def display_image(self, image, label_widget):
        try:
            img = Image.fromarray(image)
            img = img.resize((400, 400))
            imgtk = ImageTk.PhotoImage(img)
            label_widget.config(image=imgtk)
            label_widget.image = imgtk
        except Exception as e:
            print(f"Ошибка отображения: {e}")

    def import_model(self):
        path = filedialog.askopenfilename(
            initialdir=self.model_handler.MODELS_DIR,
            filetypes=[("Модели Keras", "*.keras")],
            title="Выберите файл модели"
        )
        if path:
            try:
                if not path.startswith(self.BASE_DIR):
                    raise ValueError("Модель должна находиться в папке проекта")
                self.model_handler.model = load_model(path)
                rel_path = os.path.relpath(path, self.BASE_DIR)
                self.update_model_info(rel_path)
                messagebox.showinfo("Успех", "Модель успешно загружена!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка загрузки: {str(e)}")

    def start_training(self):
        try:
            if not os.path.exists(self.DATA_DIR):
                raise FileNotFoundError(f"Папка данных не найдена: {self.DATA_DIR}")
            required_folders = ['cloudy', 'desert', 'green_area', 'water']
            for folder in required_folders:
                class_path = os.path.join(self.DATA_DIR, folder)
                if not os.path.exists(class_path):
                    raise FileNotFoundError(f"Не найдена папка класса: {folder}")
                if not os.listdir(class_path):
                    raise ValueError(f"Папка {folder} пуста")
            training_window = tk.Toplevel(self.master)
            training_window.title("Процесс обучения")
            training_window.geometry("400x150")
            progress = ttk.Progressbar(training_window, orient='horizontal', mode='indeterminate')
            progress.pack(pady=20)
            progress.start()
            
            def train_thread():
                try:
                    self.model_handler.build_unet()
                    self.model_handler.train(self.DATA_DIR, epochs=20)
                    messagebox.showinfo("Успех", "Обучение завершено!")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
                finally:
                    progress.stop()
                    training_window.destroy()
            
            threading.Thread(target=train_thread, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Ошибка данных", f"Проверьте данные:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentationApp(root)
    root.mainloop()
