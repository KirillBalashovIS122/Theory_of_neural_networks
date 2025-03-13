import pathlib

class Config:
    """
    Основной класс для хранения конфигурационных параметров приложения.

    Атрибуты:
        DATA_DIR (Path): Путь к директории с MIDI-файлами для обучения.
        SEQ_LENGTH (int): Длина входной последовательности для модели.
        BATCH_SIZE (int): Размер батча для обучения.
        EPOCHS (int): Количество эпох обучения.
        GENERATION_LENGTH (int): Количество нот для генерации.
        NOTE_DURATION (float): Длительность одной ноты в секундах.
        SOUNDFONT_PATH (str): Путь к файлу SoundFont для воспроизведения.
        MODEL_PARAMS (dict): Параметры для каждой модели.
    """
    DATA_DIR = pathlib.Path('/home/kbalashov/VS_Code/TONN/data/maestro-v2_extracted/maestro-v2.0.0/2018')
    SEQ_LENGTH = 30
    GENERATION_LENGTH = 100
    NOTE_DURATION = 0.3
    SOUNDFONT_PATH = '/usr/share/soundfonts/FluidR3_GM.sf2'
    
    MODEL_PARAMS = {
        "simple": {"epochs": 10, "batch_size": 32},
        "lstm": {"epochs": 15, "batch_size": 64},
        "gru": {"epochs": 20, "batch_size": 128}
    }

config = Config()
