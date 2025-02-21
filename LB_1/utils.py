import numpy as np
import pretty_midi
import pyfluidsynth

def load_midi_data(file_path):
    """Загружает MIDI-файл и возвращает список нот."""
    midi_data = pretty_midi.PrettyMIDI(file_path)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
    return notes

def prepare_sequences(notes, sequence_length=100):
    """Преобразует список нот в последовательности для обучения."""
    input_sequences = []
    output_sequences = []
    for i in range(len(notes) - sequence_length):
        input_sequences.append(notes[i:i+sequence_length])
        output_sequences.append(notes[i+sequence_length])
    X = np.array(input_sequences)
    y = np.array(output_sequences)
    X = X / 128.0  # Нормализация
    y = tf.keras.utils.to_categorical(y, num_classes=128)
    return X, y

def play_midi(notes):
    """Воспроизводит последовательность нот как MIDI-файл."""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start_time = 0
    for note in notes:
        end_time = start_time + 0.5  # Длительность ноты
        midi_note = pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=end_time)
        instrument.notes.append(midi_note)
        start_time = end_time
    midi.instruments.append(instrument)
    midi.write("generated_music.midi")
    fs = pyfluidsynth.Synth()
    sfid = fs.sfload("soundfont.sf2")
    fs.program_select(0, sfid, 0, 0)
    fs.start()
    fs.play_midi("generated_music.midi")
    fs.delete()
    