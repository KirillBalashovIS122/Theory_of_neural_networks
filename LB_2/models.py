"""
Модуль с реализацией архитектур нейронных сетей.
Содержит упрощенные версии автоэнкодеров.
"""
import tensorflow as tf
from tensorflow.keras import layers

class DenseAutoencoder(tf.keras.Model):
    """Упрощенный полносвязный автоэнкодер"""
    def __init__(self, latent_dim=32):
        """
        Инициализация автоэнкодера
        Args:
            latent_dim (int): Размер скрытого представления
        """
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        """
        Прямой проход данных через сеть
        Args:
            x (tf.Tensor): Входные данные
        Returns:
            tf.Tensor: Реконструированные данные
        """
        return self.decoder(self.encoder(x))

class ConvAutoencoder(tf.keras.Model):
    """Упрощенный сверточный автоэнкодер"""
    def __init__(self):
        """Инициализация сверточного автоэнкодера"""
        super().__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2))
        ])
        self.decoder = tf.keras.Sequential([
            layers.UpSampling2D((2,2)),
            layers.Conv2D(1, (3,3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        """
        Прямой проход данных через сеть
        Args:
            x (tf.Tensor): Входные данные
        Returns:
            tf.Tensor: Реконструированные данные
        """
        return self.decoder(self.encoder(x))
