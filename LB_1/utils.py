import numpy as np
import pretty_midi
import subprocess
import pygame
from config import config

def generate_music_sequence(model, seq_length, gen_length):
    """
    Генерация музыкальной последовательности с использованием модели.

    Аргументы:
        model (keras.Model): Обученная модель для генерации.
        seq_length (int): Длина входной последовательности.
        gen_length (int): Количество нот для генерации.

    Возвращает:
        list: Сгенерированная последовательность нот.
    """
    seed = np.random.randint(0, 128, size=(seq_length,))
    generated = []
    for _ in range(gen_length):
        input_seq = seed[-seq_length:].reshape(1, -1)
        pred = model.predict(input_seq, verbose=0)[0]
        next_note = np.random.choice(128, p=pred/np.sum(pred))
        generated.append(next_note)
        seed = np.append(seed, next_note)
    return generated

def save_midi(sequence, note_duration):
    """
    Сохранение последовательности в MIDI-файл.

    Аргументы:
        sequence (list): Последовательность нот.
        note_duration (float): Длительность каждой ноты в секундах.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start_time = 0.0
    for pitch in sequence:
        end_time = start_time + note_duration
        instrument.notes.append(pretty_midi.Note(
            velocity=100,
            pitch=int(pitch),
            start=start_time,
            end=end_time
        ))
        start_time = end_time
    pm.instruments.append(instrument)
    pm.write('temp.mid')

def play_midi(midi_file):
    """
    Воспроизведение MIDI-файла.

    Аргументы:
        midi_file (str): Путь к MIDI-файлу.

    Возвращает:
        subprocess.Popen: Процесс воспроизведения.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()
        return None
    except pygame.error:
        return subprocess.Popen(['fluidsynth', '-a', 'alsa', config.SOUNDFONT_PATH, midi_file])
