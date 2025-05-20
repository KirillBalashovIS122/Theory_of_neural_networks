import os
import sys
import signal
import logging
import tkinter as tk
from tkinter import messagebox
from ui import UIHandler
from model import ModelHandler

# Настройка GPU перед импортом других компонентов
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class FakeNewsClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Classifier")
        self.running = True
        
        # Настройка обработки Ctrl+C
        signal.signal(signal.SIGINT, self.safe_exit)
        
        # Инициализация компонентов
        self.model_handler = ModelHandler()
        self.ui_handler = UIHandler(root, self)
        
        # Настройка приложения
        self.training_in_progress = False
        self.setup_logging()
        
    def setup_logging(self):
        """Настройка системы логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app.log', mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Приложение инициализировано")
        
    def safe_exit(self, signum, frame):
        """Безопасный выход из приложения"""
        self.logger.info("Получен сигнал завершения")
        self.running = False
        
        if hasattr(self, 'training_in_progress') and self.training_in_progress:
            self.logger.warning("Прерывание процесса обучения")
            self.model_handler.stop_training = True
            
        if hasattr(self, 'ui_handler') and hasattr(self.ui_handler, 'progress_bar'):
            self.ui_handler.progress_bar.close()
            
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = FakeNewsClassifierApp(root)
        
        # Центрирование окна
        window_width = 800
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_right = int(screen_width / 2 - window_width / 2)
        root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")
        
        root.protocol("WM_DELETE_WINDOW", lambda: app.safe_exit(None, None))
        root.mainloop()
        
    except Exception as e:
        logging.critical(f"Критическая ошибка приложения: {e}", exc_info=True)
        messagebox.showerror("Fatal Error", f"Application crashed:\n{str(e)}")
        sys.exit(1)
