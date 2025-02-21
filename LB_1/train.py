import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from utils import load_midi_data, prepare_sequences

# Создание папки для моделей
if not os.path.exists("models"):
    os.makedirs("models")

# Загрузка данных
notes = load_midi_data("/home/kbalashov/VS_Code/TONN/data/maestro-v2_extracted/maestro-v2.0.0")
X, y = prepare_sequences(notes)

# Создание моделей
def create_model1():
    model = Sequential([
        LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def create_model2():
    model = Sequential([
        LSTM(512, input_shape=(X.shape[1], 1), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(128, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def create_model3():
    model = Sequential([
        LSTM(256, input_shape=(X.shape[1], 1), return_sequences=True),
        LSTM(128),
        Dropout(0.3),
        Dense(128, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Обучение моделей
models = {
    "model1": create_model1(),
    "model2": create_model2(),
    "model3": create_model3()
}

for name, model in models.items():
    print(f"Обучение {name}...")
    checkpoint = ModelCheckpoint(f"models/{name}.h5", monitor='loss', save_best_only=True, mode='min')
    model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint])
    print(f"{name} обучена и сохранена.")
