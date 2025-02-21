import sys
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np
import pretty_midi
import collections
import glob
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import fluidsynth
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QComboBox, QProgressBar, QFileDialog, QWidget
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Конфигурация
class Config:
    DATA_DIR = pathlib.Path('/home/kbalashov/VS_Code/TONN/data/maestro-v2_extracted/maestro-v2.0.0')
    SEED = 42
    SEQ_LENGTH = 50  
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    SOUNDFONT_PATH = "/usr/share/sounds/sf2/FluidR3_GM.sf2"  # Укажи свой путь к soundfont

config = Config()
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

# Функция загрузки всех MIDI-файлов
def load_all_midi_files(data_dir: pathlib.Path):
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    return filenames if filenames else []

# Функция преобразования MIDI в DataFrame
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    
    for note in instrument.notes:
        notes['pitch'].append(note.pitch)
        notes['start'].append(note.start)
        notes['end'].append(note.end)
        notes['duration'].append(note.end - note.start)
    
    return pd.DataFrame(notes)

# Подготовка данных для RNN из всех файлов
def prepare_dataset_from_folder(data_dir: pathlib.Path, seq_length: int):
    all_notes = []
    midi_files = load_all_midi_files(data_dir)
    
    for midi_file in midi_files:
        notes = midi_to_notes(midi_file)
        all_notes.extend(notes['pitch'].values)
    
    input_seqs, target_seqs = [], []
    
    for i in range(len(all_notes) - seq_length):
        input_seqs.append(all_notes[i:i + seq_length])
        target_seqs.append(all_notes[i + seq_length])
    
    return np.array(input_seqs), np.array(target_seqs)

# Построение RNN-модели
def build_model(hidden_units):
    model = Sequential([
        layers.Embedding(input_dim=128, output_dim=64, input_length=config.SEQ_LENGTH),
        layers.LSTM(hidden_units, return_sequences=True),
        layers.LSTM(hidden_units),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), metrics=['accuracy'])
    return model

# Функция генерации нот
def generate_music(model, seed_sequence, length=100):
    generated = list(seed_sequence)
    for _ in range(length):
        input_seq = np.array(generated[-config.SEQ_LENGTH:]).reshape(1, -1)
        prediction = np.argmax(model.predict(input_seq), axis=-1)[0]
        generated.append(prediction)
    return generated

# Функция сохранения сгенерированной музыки в MIDI
def save_generated_midi(notes, out_file="generated_music.mid"):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    
    start_time = 0.0
    for pitch in notes:
        note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=start_time, end=start_time + 0.5)
        instrument.notes.append(note)
        start_time += 0.5
    
    pm.instruments.append(instrument)
    pm.write(out_file)
    return out_file

# Воспроизведение MIDI
def play_midi(midi_file):
    fs = fluidsynth.Synth()
    fs.start()
    sfid = fs.sfload(config.SOUNDFONT_PATH)
    fs.program_select(0, sfid, 0, 0)
    pm = pretty_midi.PrettyMIDI(midi_file)
    audio_data = pm.fluidsynth()
    fs.write_audio("temp.wav", audio_data)
    fs.delete()

# Интерфейс PyQt5
class MusicGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Generator")
        self.setGeometry(100, 100, 800, 600)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.model_label = QLabel("Выберите модель")
        self.layout.addWidget(self.model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["64 units", "128 units", "256 units"])
        self.layout.addWidget(self.model_combo)
        
        self.train_button = QPushButton("Обучить модель")
        self.train_button.clicked.connect(self.train_model)
        self.layout.addWidget(self.train_button)
        
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        
        self.generate_button = QPushButton("Сгенерировать музыку")
        self.generate_button.clicked.connect(self.generate_music)
        self.layout.addWidget(self.generate_button)
        
        self.play_button = QPushButton("Проиграть музыку")
        self.play_button.clicked.connect(self.play_music)
        self.layout.addWidget(self.play_button)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.models = {}
        self.current_model = None
        self.generated_midi = None
    
    def train_model(self):
        self.X_train, self.y_train = prepare_dataset_from_folder(config.DATA_DIR, config.SEQ_LENGTH)
        split_idx = int(0.8 * len(self.X_train))
        self.X_test, self.y_test = self.X_train[split_idx:], self.y_train[split_idx:]
        self.X_train, self.y_train = self.X_train[:split_idx], self.y_train[:split_idx]

        units = int(self.model_combo.currentText().split()[0])
        self.current_model = build_model(units)
        
        self.progress_bar.setMaximum(config.EPOCHS)
        self.progress_bar.setValue(0)
        
        def update_progress(epoch, logs):
            self.progress_bar.setValue(epoch + 1)
            QApplication.processEvents()
        
        self.current_model.fit(self.X_train, self.y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_data=(self.X_test, self.y_test), callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=update_progress)])
        self.models[units] = self.current_model
    
    def generate_music(self):
        seed_seq = self.X_test[0]
        generated_sequence = generate_music(self.current_model, seed_seq)
        self.generated_midi = save_generated_midi(generated_sequence)
    
    def play_music(self):
        if self.generated_midi:
            play_midi(self.generated_midi)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicGeneratorApp()
    window.show()
    sys.exit(app.exec_())
