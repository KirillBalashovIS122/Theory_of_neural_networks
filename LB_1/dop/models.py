import tensorflow as tf
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

SEQ_LENGTH = 30
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

def get_next_version(model_type):
    version = 1
    while True:
        filename = f"{model_type}_v{version}.keras"
        if not os.path.exists(filename):
            return version
        version += 1

def save_model_with_version(model, model_type):
    version = get_next_version(model_type)
    filename = f"{model_type}_v{version}.keras"
    model.save(filename)
    print(f"Модель сохранена как {filename}")
    return filename

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

def train_model(model, model_type, X, y, epochs, batch_size):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=callbacks, verbose=1)
    saved_model_path = save_model_with_version(model, model_type)
    return history, saved_model_path

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
