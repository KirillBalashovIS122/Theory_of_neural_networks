"""
Модуль с реализацией архитектур нейронных сетей.
Содержит упрощенные версии автоэнкодеров.
"""
import tensorflow as tf
from tensorflow.keras import layers
import logging

class DenseAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim=32):
        super().__init__()
        logging.debug(f"Создание DenseAutoencoder с latent_dim={latent_dim}")
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])
        logging.info("DenseAutoencoder успешно инициализирован")

    def call(self, x):
        logging.debug(f"Вызов DenseAutoencoder с входными данными формы {x.shape}")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logging.debug(f"Выход DenseAutoencoder формы {decoded.shape}")
        return decoded

class ConvAutoencoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        logging.debug("Создание ConvAutoencoder")
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2))
        ])
        self.decoder = tf.keras.Sequential([
            layers.UpSampling2D((2,2)),
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
        ])
        logging.info("ConvAutoencoder успешно инициализирован")

    def call(self, x):
        logging.debug(f"Вызов ConvAutoencoder с входными данными формы {x.shape}")
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        logging.debug(f"Выход ConvAutoencoder формы {decoded.shape}")
        return decoded
