import collections
import os
import numpy as np
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt

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

def plot_piano_roll(notes_df, title="Сгенерированные ноты"):
    if notes_df.empty:
        return None
    
    plt.figure(figsize=(12, 6))
    
    if 'start' in notes_df.columns:
        start_times = notes_df['start'].values
    else:
        times = np.zeros(len(notes_df))
        for i in range(1, len(notes_df)):
            times[i] = times[i-1] + notes_df['step'].iloc[i-1]
        start_times = times
    
    pitches = notes_df['pitch'].values
    durations = notes_df['duration'].values
    
    for i in range(len(notes_df)):
        plt.plot([start_times[i], start_times[i] + durations[i]], 
                 [pitches[i], pitches[i]], 
                 linewidth=1, color='blue')
    
    plt.scatter(start_times, pitches, color='red', s=30, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Время (секунды)')
    plt.ylabel('MIDI номер ноты')
    plt.grid(True, alpha=0.3)
    
    y_min = max(0, min(pitches) - 12)
    y_max = min(127, max(pitches) + 12)
    plt.ylim(y_min, y_max)
    
    c_notes = [24, 36, 48, 60, 72, 84, 96, 108]
    c_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']
    plt.yticks(c_notes, c_labels)
    
    plt.tight_layout()
    return plt.gcf()
