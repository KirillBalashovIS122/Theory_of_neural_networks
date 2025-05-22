import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from model import SegmentationModel

class SegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Сегментация животных")
        self.master.geometry("1200x800")
        self.model = SegmentationModel()
        self.current_image = None
        self.result_image = None
        self.model_loaded = False
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        try:
            main_frame = tk.Frame(self.master)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            left_frame = tk.Frame(main_frame)
            left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            right_frame = tk.Frame(main_frame, width=350)
            right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

            self.img_frame = tk.LabelFrame(left_frame, text="Изображения")
            self.img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.original_img = tk.Label(self.img_frame, bg='white')
            self.original_img.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.result_img = tk.Label(self.img_frame, bg='white')
            self.result_img.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

            control_frame = tk.Frame(left_frame)
            control_frame.pack(fill=tk.X, pady=10)
            
            ttk.Button(control_frame, text="Загрузить изображение", 
                     command=self.load_image).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Обучить модель", 
                     command=self.start_training).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Сегментировать", 
                     command=self.segment_image).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Сохранить результат", 
                     command=self.save_result).pack(side=tk.LEFT, padx=5)
            ttk.Button(control_frame, text="Выход", 
                     command=self.master.quit).pack(side=tk.RIGHT, padx=5)

            settings_frame = tk.LabelFrame(right_frame, text="Настройки", padx=10, pady=10)
            settings_frame.pack(fill=tk.BOTH, padx=5, pady=5)
            
            tk.Label(settings_frame, text="Настройка цветов:", pady=10).pack(anchor='w')
            self.color_btns = {}
            color_frame = tk.Frame(settings_frame)
            color_frame.pack(fill=tk.X)
            
            for class_id in self.model.CLASS_NAMES:
                btn_frame = tk.Frame(color_frame)
                btn_frame.pack(fill=tk.X, pady=2)
                
                tk.Label(btn_frame, text=self.model.CLASS_NAMES[class_id], width=10).pack(side=tk.LEFT)
                btn = tk.Button(btn_frame, bg=self.model.rgb_to_hex(self.model.CLASS_COLORS[class_id]),
                               width=3, command=lambda c=class_id: self.change_color(c))
                btn.pack(side=tk.LEFT)
                self.color_btns[class_id] = btn

            info_frame = tk.LabelFrame(right_frame, text="Информация", padx=10, pady=10)
            info_frame.pack(fill=tk.BOTH, padx=5, pady=5)
            
            self.image_info = tk.Label(info_frame, text="Изображение: Отсутствует", anchor='w')
            self.image_info.pack(fill=tk.X)
            
            self.status_info = tk.Label(info_frame, text="Статус: Готов", anchor='w')
            self.status_info.pack(fill=tk.X)
        except Exception as e:
            print(f"Ошибка инициализации интерфейса: {str(e)}")

    def load_model(self):
        self.status_info.config(text="Статус: Загрузка модели...")
        self.master.update()
        
        def load():
            try:
                if self.model.load_pretrained_model():
                    self.model_loaded = True
                    print("Модель успешно загружена и готова к сегментации")
                else:
                    print("Ошибка: Не удалось загрузить модель")
                    messagebox.showerror("Ошибка", "Не удалось загрузить модель!")
            except Exception as e:
                print(f"Ошибка загрузки модели: {str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка загрузки модели: {str(e)}")
            finally:
                self.status_info.config(text="Статус: Готов")
        
        threading.Thread(target=load).start()

    def load_image(self):
        try:
            path = filedialog.askopenfilename(
                filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp")])
            if path:
                self.current_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                self.display_image(self.current_image, self.original_img)
                self.image_info.config(text=f"Изображение: {os.path.basename(path)}")
                self.result_image = None
                self.result_img.config(image='')
                print(f"Изображение {os.path.basename(path)} успешно загружено")
        except Exception as e:
            print(f"Ошибка загрузки изображения: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка загрузки изображения: {str(e)}")

    def start_training(self):
        try:
            if self.model_loaded:
                res = messagebox.askyesno("Подтверждение", 
                    "Переобучить существующую модель? Данные обучения будут утеряны!")
                if not res:
                    return

            dataset_path = filedialog.askdirectory(title="Выберите папку с данными")
            if not dataset_path:
                return

            self.status_info.config(text="Статус: Обучение модели...")
            
            def train():
                try:
                    self.model.train_custom_model(dataset_path)
                    self.model_loaded = True
                    messagebox.showinfo("Успех", "Обучение завершено!")
                    print("Модель успешно обучена и готова к сегментации")
                except Exception as e:
                    print(f"Ошибка обучения модели: {str(e)}")
                    messagebox.showerror("Ошибка", f"Ошибка обучения модели: {str(e)}")
                finally:
                    self.status_info.config(text="Статус: Готов")
            
            threading.Thread(target=train).start()
        except Exception as e:
            print(f"Ошибка запуска обучения: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка запуска обучения: {str(e)}")

    def segment_image(self):
        try:
            if not self.model_loaded:
                print("Ошибка: Модель не загружена")
                messagebox.showerror("Ошибка", "Модель не загружена!")
                return
            if self.current_image is None:
                print("Ошибка: Изображение не загружено")
                messagebox.showerror("Ошибка", "Изображение не загружено!")
                return
            
            self.status_info.config(text="Статус: Сегментация...")
            
            def segment():
                try:
                    self.result_image = self.model.predict(
                        self.current_image,
                        list(self.model.CLASS_NAMES.keys()),
                        transparency=1.0
                    )
                    self.display_image(self.result_image, self.result_img)
                    print("Сегментация успешно выполнена")
                except Exception as e:
                    print(f"Ошибка сегментации: {str(e)}")
                    messagebox.showerror("Ошибка", f"Ошибка сегментации: {str(e)}")
                finally:
                    self.status_info.config(text="Статус: Готов")
            
            threading.Thread(target=segment).start()
        except Exception as e:
            print(f"Ошибка запуска сегментации: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка запуска сегментации: {str(e)}")

    def save_result(self):
        try:
            if self.result_image is None:
                print("Ошибка: Нет результатов для сохранения")
                messagebox.showerror("Ошибка", "Нет результатов для сохранения!")
                return
                
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
            if path:
                cv2.imwrite(path, cv2.cvtColor(self.result_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Успех", "Результат сохранён")
                print(f"Результат сохранён как {path}")
        except Exception as e:
            print(f"Ошибка сохранения результата: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка сохранения результата: {str(e)}")

    def change_color(self, class_id):
        try:
            color = simpledialog.askstring(
                "Выбор цвета",
                f"Введите HEX-цвет для {self.model.CLASS_NAMES[class_id]}:",
                initialvalue=self.model.rgb_to_hex(self.model.CLASS_COLORS[class_id]))
            
            if color and self.model.is_hex_color(color):
                self.model.CLASS_COLORS[class_id] = self.model.hex_to_rgb(color)
                self.color_btns[class_id].config(bg=color)
                if self.result_image is not None:
                    self.segment_image()
                print(f"Цвет для {self.model.CLASS_NAMES[class_id]} изменён на {color}")
            else:
                print(f"Ошибка: Неверный формат HEX-цвета: {color}")
                messagebox.showerror("Ошибка", "Неверный формат HEX-цвета!")
        except Exception as e:
            print(f"Ошибка изменения цвета: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка изменения цвета: {str(e)}")

    def display_image(self, image, label_widget):
        try:
            img = Image.fromarray(image)
            max_size = (550, 550)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)
            label_widget.config(image=imgtk)
            label_widget.image = imgtk
        except Exception as e:
            print(f"Ошибка отображения изображения: {str(e)}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SegmentationApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Ошибка запуска приложения: {str(e)}")