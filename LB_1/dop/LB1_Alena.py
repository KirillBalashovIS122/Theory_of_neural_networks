import collections
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import glob
from tqdm import tqdm
import pathlib

# Настройки модели
SEQ_LENGTH = 30
EPOCHS = 5
BATCH_SIZE = 32
NUM_NOTES = 50

# Пути к данным
DATA_DIR = pathlib.Path("/home/kbalashov/VS Code/TONN/data/maestro-v2_extracted/maestro-v2.0.0/2018")

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU доступен и будет использоваться.")
    else:
        print("GPU не доступен, используется CPU.")

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start if sorted_notes else 0
    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start
    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        midi_note = pretty_midi.Note(velocity=velocity, pitch=int(note['pitch']), start=start, end=end)
        instrument.notes.append(midi_note)
        prev_start = start
    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

def normalize_data(notes_df):
    scaler_pitch = MinMaxScaler()
    scaler_step = MinMaxScaler()
    scaler_duration = MinMaxScaler()
    notes_df['pitch'] = scaler_pitch.fit_transform(notes_df[['pitch']])
    notes_df['step'] = scaler_step.fit_transform(notes_df[['step']])
    notes_df['duration'] = scaler_duration.fit_transform(notes_df[['duration']])
    return notes_df, scaler_pitch, scaler_step, scaler_duration

def create_sequences(notes_df, seq_length=SEQ_LENGTH):
    X, y = [], []
    for i in tqdm(range(len(notes_df) - seq_length), desc="Создание последовательностей"):
        X.append(notes_df.iloc[i:i+seq_length][['pitch', 'step', 'duration']].values)
        y.append(notes_df.iloc[i+seq_length][['pitch', 'step', 'duration']].values)
    return np.array(X), np.array(y)

def create_model1():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        SimpleRNN(64),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, "simple_rnn"

def create_model2():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(3)
    ])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='mse')
    return model, "lstm"

def create_model3():
    model = Sequential([
        Input(shape=(SEQ_LENGTH, 3)),
        GRU(256, return_sequences=True),
        GRU(128),
        Dense(64, activation='tanh'),
        Dense(3)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model, "gru"

def train_model(model, X, y, epochs, batch_size):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks, verbose=1)
    return history

def generate_music(model, initial_notes, num_notes, scalers):
    generated = initial_notes.copy()
    if len(generated) < SEQ_LENGTH:
        raise ValueError(f"Начальная последовательность должна быть не менее {SEQ_LENGTH} нот")
    for _ in tqdm(range(num_notes), desc="Генерация нот"):
        input_seq = generated[-SEQ_LENGTH:]
        input_seq = np.expand_dims(input_seq, axis=0)
        predicted = model.predict(input_seq, verbose=0)[0]
        generated = np.vstack([generated, predicted])
    scaler_pitch, scaler_step, scaler_duration = scalers
    try:
        generated[:, 0] = scaler_pitch.inverse_transform(generated[:, 0].reshape(-1, 1)).flatten()
        generated[:, 1] = scaler_step.inverse_transform(generated[:, 1].reshape(-1, 1)).flatten()
        generated[:, 2] = scaler_duration.inverse_transform(generated[:, 2].reshape(-1, 1)).flatten()
    except Exception as e:
        print(f"Ошибка денормализации: {e}")
        return pd.DataFrame()
    return pd.DataFrame(generated, columns=['pitch', 'step', 'duration'])

def plot_training_history(histories, models_to_process):
    # Если выбрана одна модель, отображаем только её график
    if len(models_to_process) == 1:
        model_name = models_to_process[0]
        plt.figure(figsize=(12, 6))
        history = histories[model_name]
        plt.plot(history.history['loss'], label="Обучающая потеря")
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label="Валидационная потеря")
        plt.title(f"График обучения модели {model_name}")
        plt.xlabel("Эпоха")
        plt.ylabel("Потеря")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # Для всех моделей создаём три подграфика
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        model_order = ['SimpleRNN', 'LSTM', 'GRU']
        
        for i, model_name in enumerate(model_order):
            ax = axes[i]
            if model_name in histories:
                history = histories[model_name]
                ax.plot(history.history['loss'], label="Обучающая потеря")
                if 'val_loss' in history.history:
                    ax.plot(history.history['val_loss'], label="Валидационная потеря")
                ax.set_title(f"График обучения для модели {model_name}")
                ax.set_ylabel("Потеря")
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"Нет данных для {model_name}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"График обучения для модели {model_name}")
                ax.set_ylabel("Потеря")
        
        axes[-1].set_xlabel("Эпоха")
        plt.tight_layout()
        plt.show()

def plot_piano_rolls(generated_notes_dict, models_to_process):
    colors = {'SimpleRNN': 'blue', 'LSTM': 'green', 'GRU': 'purple'}
    
    # Если выбрана одна модель, отображаем только её график
    if len(models_to_process) == 1:
        model_key = models_to_process[0]
        plt.figure(figsize=(12, 6))
        if model_key in generated_notes_dict and not generated_notes_dict[model_key].empty:
            notes_df = generated_notes_dict[model_key]
            
            if 'start' in notes_df.columns:
                start_times = notes_df['start'].values
            else:
                times = np.zeros(len(notes_df))
                for j in range(1, len(notes_df)):
                    times[j] = times[j-1] + notes_df['step'].iloc[j-1]
                start_times = times
            
            pitches = notes_df['pitch'].values
            durations = notes_df['duration'].values
            
            for k in range(len(notes_df)):
                plt.plot([start_times[k], start_times[k] + durations[k]], 
                         [pitches[k], pitches[k]], 
                         linewidth=1, color=colors[model_key], alpha=0.7)
            
            plt.scatter(start_times, pitches, color=colors[model_key], s=30, alpha=0.8, label=model_key)
            
            plt.title(f"Сгенерированные ноты моделью {model_key}")
            plt.xlabel('Время (секунды)')
            plt.ylabel('MIDI номер ноты')
            plt.grid(True, alpha=0.3)
            
            all_pitches = pitches
            if len(all_pitches) > 0:
                y_min = max(0, min(all_pitches) - 12)
                y_max = min(127, max(all_pitches) + 12)
                plt.ylim(y_min, y_max)
            
            c_notes = [24, 36, 48, 60, 72, 84, 96, 108]
            c_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
            plt.yticks(c_notes, c_labels)
            
            plt.legend()
        else:
            plt.text(0.5, 0.5, f"Нет данных для {model_key}", 
                     horizontalalignment='center', verticalalignment='center')
            plt.title(f"Сгенерированные ноты моделью {model_key}")
            plt.xlabel('Время (секунды)')
            plt.ylabel('MIDI номер ноты')
        
        plt.tight_layout()
        plt.show()
    else:
        # Для всех моделей создаём три подграфика
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        model_order = ['SimpleRNN', 'LSTM', 'GRU']
        
        for idx, model_key in enumerate(model_order):
            ax = axes[idx]
            if model_key in generated_notes_dict and not generated_notes_dict[model_key].empty:
                notes_df = generated_notes_dict[model_key]
                
                if 'start' in notes_df.columns:
                    start_times = notes_df['start'].values
                else:
                    times = np.zeros(len(notes_df))
                    for j in range(1, len(notes_df)):
                        times[j] = times[j-1] + notes_df['step'].iloc[j-1]
                    start_times = times
                
                pitches = notes_df['pitch'].values
                durations = notes_df['duration'].values
                
                for k in range(len(notes_df)):
                    ax.plot([start_times[k], start_times[k] + durations[k]], 
                            [pitches[k], pitches[k]], 
                            linewidth=1, color=colors[model_key], alpha=0.7)
                
                ax.scatter(start_times, pitches, color=colors[model_key], s=30, alpha=0.8, label=model_key)
                
                ax.set_title(f"Сгенерированные ноты моделью {model_key}")
                ax.set_ylabel('MIDI номер ноты')
                ax.grid(True, alpha=0.3)
                
                all_pitches = pitches
                if len(all_pitches) > 0:
                    y_min = max(0, min(all_pitches) - 12)
                    y_max = min(127, max(all_pitches) + 12)
                    ax.set_ylim(y_min, y_max)
                
                c_notes = [24, 36, 48, 60, 72, 84, 96, 108]
                c_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
                ax.set_yticks(c_notes)
                ax.set_yticklabels(c_labels)
                
                ax.legend()
            else:
                ax.text(0.5, 0.5, f"Нет данных для {model_key}", 
                        horizontalalignment='center', verticalalignment='center')
                ax.set_title(f"Сгенерированные ноты моделью {model_key}")
                ax.set_ylabel('MIDI номер ноты')
        
        axes[-1].set_xlabel('Время (секунды)')
        plt.tight_layout()
        plt.show()

def generate_and_save_music(model, model_name, model_type, initial_sequence, scalers):
    generated_notes = generate_music(model, initial_sequence, NUM_NOTES, scalers)
    if not generated_notes.empty:
        output_file = f"generated_{model_type}.mid"
        notes_to_midi(generated_notes, output_file, "Acoustic Grand Piano")
        print(f"Музыка сгенерирована моделью {model_name} и сохранена в {output_file}")
        return generated_notes
    else:
        print(f"Ошибка: Не удалось сгенерировать музыку с помощью модели {model_name}.")
        return pd.DataFrame()

def main():
    check_gpu()
    
    # Загрузка данных
    try:
        filenames = glob.glob(str(DATA_DIR/'**/*.mid*'), recursive=True)
        all_notes = []
        for f in tqdm(filenames[:10], desc="Загрузка MIDI файлов"):
            all_notes.append(midi_to_notes(f))
        notes_df = pd.concat(all_notes)
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        exit()
    
    # Нормализация данных
    notes_df, scaler_pitch, scaler_step, scaler_duration = normalize_data(notes_df)
    
    # Создание последовательностей
    X, y = create_sequences(notes_df)
    
    # Определение моделей
    models = {
        "SimpleRNN": create_model1(),
        "LSTM": create_model2(),
        "GRU": create_model3()
    }
    
    # Выбор модели(ей)
    print("Выберите модель для обучения и генерации:")
    print("1. SimpleRNN")
    print("2. LSTM")
    print("3. GRU")
    print("4. Все модели")
    model_choice = input("Введите номер модели (1-4): ")
    
    histories = {}
    generated_notes_dict = {}
    initial_sequence = X[0].copy()
    scalers = (scaler_pitch, scaler_step, scaler_duration)
    
    if model_choice == "4":
        models_to_process = ["SimpleRNN", "LSTM", "GRU"]
        # Последовательное обучение всех моделей
        for model_name in models_to_process:
            model, _ = models[model_name]
            print(f"\nОбучение модели {model_name} (эпох: {EPOCHS}, батч: {BATCH_SIZE})")
            history = train_model(model, X, y, EPOCHS, BATCH_SIZE)
            histories[model_name] = history
        
        # Последовательная генерация музыки всеми моделями
        for model_name in models_to_process:
            model, model_type = models[model_name]
            generated_notes = generate_and_save_music(model, model_name, model_type, initial_sequence, scalers)
            generated_notes_dict[model_name] = generated_notes
    else:
        if model_choice == "1":
            models_to_process = ["SimpleRNN"]
        elif model_choice == "2":
            models_to_process = ["LSTM"]
        elif model_choice == "3":
            models_to_process = ["GRU"]
        else:
            print("Неверный выбор. Выход из программы.")
            exit()
        
        model_name = models_to_process[0]
        model, model_type = models[model_name]
        print(f"\nОбучение модели {model_name} (эпох: {EPOCHS}, батч: {BATCH_SIZE})")
        history = train_model(model, X, y, EPOCHS, BATCH_SIZE)
        histories[model_name] = history
        generated_notes = generate_and_save_music(model, model_name, model_type, initial_sequence, scalers)
        generated_notes_dict[model_name] = generated_notes
    
    # Отображение графика обучения
    plot_training_history(histories, models_to_process)
    
    # Отображение графиков сгенерированных нот
    plot_piano_rolls(generated_notes_dict, models_to_process)

if __name__ == "__main__":
    main()
