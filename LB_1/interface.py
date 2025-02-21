import sys
import ipywidgets as widgets
from IPython.display import display, clear_output
from train import models
from generate import generate_music
from utils import play_midi

# Проверяем, запущен ли код в Jupyter Notebook
try:
    get_ipython()
    IN_JUPYTER = True
except NameError:
    IN_JUPYTER = False

# Выбор модели
model_selector = widgets.Dropdown(
    options=list(models.keys()),
    description="Модель:",
    disabled=False
)

# Поле для ввода длины генерируемой последовательности
length_slider = widgets.IntSlider(
    value=500,
    min=100,
    max=1000,
    step=100,
    description="Длина:",
    disabled=False
)

# Поле для ввода начальной последовательности
seed_text = widgets.Text(
    value="60, 62, 64, 65",  # Пример начальной последовательности
    description="Начальная последовательность:",
    disabled=False
)

# Поле для ввода температуры
temperature_slider = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=2.0,
    step=0.1,
    description="Температура:",
    disabled=False
)

# Кнопка для генерации музыки
generate_button = widgets.Button(description="Сгенерировать музыку")

# Поле для вывода результата
output = widgets.Output()

# Обработчик нажатия кнопки
def on_generate_button_clicked(b):
    with output:
        clear_output()
        print("Генерация музыки...")

        # Получаем выбранную модель
        selected_model = models[model_selector.value]

        # Преобразуем начальную последовательность в список нот
        seed_sequence = list(map(int, seed_text.value.split(',')))

        # Генерация музыки с учетом температуры
        generated_music = generate_music(selected_model, seed_sequence, length_slider.value, temperature_slider.value)

        # Воспроизведение музыки
        print("Воспроизведение музыки...")
        play_midi(generated_music)
        print("Готово!")

# Привязка обработчика к кнопке
generate_button.on_click(on_generate_button_clicked)

if __name__ == "__main__":
    if IN_JUPYTER:
        # Если код выполняется в Jupyter, используем display()
        display(model_selector, length_slider, seed_text, temperature_slider, generate_button, output)
    else:
        # Если код запускается в терминале VS Code, работаем через консоль
        print("Выберите модель:")
        for i, key in enumerate(models.keys()):
            print(f"{i + 1}. {key}")

        model_choice = int(input("Введите номер модели: ")) - 1
        model_name = list(models.keys())[model_choice]
        selected_model = models[model_name]

        seed_sequence = input("Введите начальную последовательность нот через запятую (например, 60, 62, 64, 65): ")
        seed_sequence = list(map(int, seed_sequence.split(',')))

        length = int(input("Введите длину генерируемой последовательности (100-1000): "))
        temperature = float(input("Введите температуру (0.1-2.0): "))

        print("Генерация музыки...")
        generated_music = generate_music(selected_model, seed_sequence, length, temperature)

        print("Воспроизведение музыки...")
        play_midi(generated_music)

        print("Готово! Файл `generated_music.midi` сохранен.")
