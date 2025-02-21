import numpy as np
from tensorflow.keras.models import load_model
from utils import play_midi

def generate_music(model, seed_sequence, length=500, temperature=1.0):
    """Генерирует музыку с помощью модели."""
    generated_sequence = seed_sequence.copy()
    for _ in range(length):
        # Предсказание следующей ноты с учетом температуры
        predicted_note = model.predict(np.array([generated_sequence[-100:]]))
        predicted_note = np.argmax(predicted_note / temperature)
        generated_sequence.append(predicted_note)
    return generated_sequence

# Пример использования
if __name__ == "__main__":
    model = load_model("models/model1.h5")
    seed_sequence = [60, 62, 64, 65]  # Начальная последовательность
    temperature = 1.5  # Параметр температуры для управления случайностью
    generated_music = generate_music(model, seed_sequence, length=500, temperature=temperature)
    play_midi(generated_music)
    