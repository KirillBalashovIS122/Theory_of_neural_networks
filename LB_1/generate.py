import numpy as np
from tensorflow.keras.models import load_model
from utils import play_midi

def generate_music(model, seed_sequence, length=500):
    """Генерирует музыку с помощью модели."""
    generated_sequence = seed_sequence.copy()
    for _ in range(length):
        predicted_note = model.predict(np.array([generated_sequence[-100:]]))
        predicted_note = np.argmax(predicted_note)
        generated_sequence.append(predicted_note)
    return generated_sequence

# Пример использования
if __name__ == "__main__":
    model = load_model("models/model1.h5")
    seed_sequence = [60, 62, 64, 65]  # Начальная последовательность
    generated_music = generate_music(model, seed_sequence, length=500)
    play_midi(generated_music)
