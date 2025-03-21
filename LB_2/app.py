"""
Главный модуль приложения для шумоподавления изображений.
Запускает графический интерфейс пользователя.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tkinter as tk
from gui import DenoisingAppGUI

if __name__ == "__main__":
    app = DenoisingAppGUI()
    app.mainloop()
