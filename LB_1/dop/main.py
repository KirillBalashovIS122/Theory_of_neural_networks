import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

from utils import midi_to_notes, notes_to_midi, plot_piano_roll
from models import (
    check_gpu, normalize_data, create_sequences, 
    create_model1, create_model2, create_model3,
    train_model, generate_music
)
from config import SEQ_LENGTH, EPOCHS, BATCH_SIZE, NUM_NOTES, DATA_DIR

def load_data():
    try:
        filenames = glob.glob(str(DATA_DIR/'**/*.mid*'), recursive=True)
        all_notes = []
        for f in tqdm(filenames[:10], desc="Загрузка MIDI файлов"):
            all_notes.append(midi_to_notes(f))
        return pd.concat(all_notes)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        exit()

def plot_training_history(histories, models_to_plot):
    num_models = len(models_to_plot)
    if num_models == 0:
        print("Нет данных для отображения графиков.")
        return
    
    fig, axs = plt.subplots(num_models, 1, figsize=(12, 5 * num_models), sharex=True)
    
    if num_models == 1:
        axs = [axs]
    
    model_names = {"1": "SimpleRNN", "2": "LSTM", "3": "GRU"}
    
    for i, model_key in enumerate(models_to_plot):
        if model_key in histories:
            history = histories[model_key]
            ax = axs[i]
            ax.plot(history.history['loss'], label="Обучающая потеря")
            if 'val_loss' in history.history:
                ax.plot(history.history['val_loss'], label="Валидационная потеря")
            ax.set_title(f"График обучения для модели {model_names.get(model_key, model_key)}")
            ax.set_ylabel("Потеря")
            ax.legend()
            
            if i == len(models_to_plot) - 1:
                ax.set_xlabel("Эпоха")
    
    plt.tight_layout()
    plt.show()

def generate_and_save_music(model, model_name, model_type, initial_sequence, scalers):
    generated_notes = generate_music(model, initial_sequence, NUM_NOTES, scalers)
    if not generated_notes.empty:
        output_file = f"generated_{model_type}.mid"
        notes_to_midi(generated_notes, output_file, "Acoustic Grand Piano")
        print(f"Музыка сгенерирована моделью {model_name} и сохранена в {output_file}")
        
        # Убрана строка plt.figure(figsize=(12, 6)) - теперь будет только одна фигура
        plot_fig = plot_piano_roll(generated_notes, f"Ноты, сгенерированные моделью {model_name}")
        if plot_fig:
            plt.show()
        
        return generated_notes
    else:
        print(f"Ошибка: Не удалось сгенерировать музыку с помощью модели {model_name}.")
        return None

def main():
    check_gpu()
    
    notes_df = load_data()
    notes_df, scaler_pitch, scaler_step, scaler_duration = normalize_data(notes_df)
    X, y = create_sequences(notes_df)
    models = {
        "1": create_model1(),
        "2": create_model2(),
        "3": create_model3()
    }
    histories = {}
    model_paths = {}
    generated_notes = None

    while True:
        print("\n=== Генерация музыки с помощью нейронных сетей ===")
        print("1. Обучить модель")
        print("2. Сгенерировать музыку")
        print("3. Показать графики обучения")
        print("4. Визуализировать сгенерированные ноты")
        print("5. Выход")
        choice = input("Выберите действие (1-5): ")

        if choice == "1":
            print("\nВыберите модель для обучения:")
            print("1. SimpleRNN (модель 1)")
            print("2. LSTM (модель 2)")
            print("3. GRU (модель 3)")
            print("4. Все модели")
            model_choice = input("Введите номер модели (1-4): ")

            models_to_train = ["1", "2", "3"] if model_choice == "4" else [model_choice]
            for m in models_to_train:
                if m in models:
                    model, model_type = models[m]
                    print(f"\nОбучение модели {model_type} (эпох: {EPOCHS}, батч: {BATCH_SIZE})")
                    history, model_path = train_model(model, model_type, X, y, EPOCHS, BATCH_SIZE)
                    histories[m] = history
                    model_paths[m] = model_path
                else:
                    print(f"Модель {m} не найдена.")
            
            if model_choice == "4":
                plot_training_history(histories, models_to_train)

        elif choice == "2":
            print("\nВыберите модель для генерации:")
            print("1. SimpleRNN (модель 1)")
            print("2. LSTM (модель 2)")
            print("3. GRU (модель 3)")
            print("4. Все модели")
            model_choice = input("Введите номер модели (1-4): ")
                
            initial_sequence = X[0].copy()
            scalers = (scaler_pitch, scaler_step, scaler_duration)
            
            if model_choice == "4":
                print("Генерация музыки всеми моделями...")
                for m_key, (model, model_type) in models.items():
                    model_name = {"1": "SimpleRNN", "2": "LSTM", "3": "GRU"}[m_key]
                    generated = generate_and_save_music(model, model_name, model_type, initial_sequence, scalers)
                    if m_key == "1":
                        generated_notes = generated
            elif model_choice in models:
                model, model_type = models[model_choice]
                model_name = {"1": "SimpleRNN", "2": "LSTM", "3": "GRU"}[model_choice]
                generated_notes = generate_and_save_music(model, model_name, model_type, initial_sequence, scalers)
            else:
                print("Неверный выбор модели.")

        elif choice == "3":
            print("\nВыберите модель для отображения графика:")
            print("1. SimpleRNN (модель 1)")
            print("2. LSTM (модель 2)")
            print("3. GRU (модель 3)")
            print("4. Все модели")
            model_choice = input("Введите номер модели (1-4): ")
            
            if model_choice == "4":
                models_to_plot = [m for m in ["1", "2", "3"] if m in histories]
                if not models_to_plot:
                    print("Нет обученных моделей для отображения графиков.")
                else:
                    plot_training_history(histories, models_to_plot)
            elif model_choice in ["1", "2", "3"]:
                if model_choice in histories:
                    plot_training_history(histories, [model_choice])
                else:
                    print(f"История обучения для модели {model_choice} не найдена. Сначала обучите модель.")
            else:
                print("Неверный выбор. Пожалуйста, выберите 1, 2, 3 или 4.")
                
        elif choice == "4":
            if generated_notes is not None and not generated_notes.empty:
                plot_fig = plot_piano_roll(generated_notes, "Визуализация сгенерированных нот")
                if plot_fig:
                    plt.show()
            else:
                print("Сначала сгенерируйте музыку (опция 2).")

        elif choice == "5":
            print("Выход из программы.")
            break

        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()
