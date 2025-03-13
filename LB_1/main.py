import sys
from PyQt5.QtWidgets import QApplication
from ui import MusicGeneratorUI

def main():
    """Основная функция запуска приложения."""
    app = QApplication(sys.argv)
    window = MusicGeneratorUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
