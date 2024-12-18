import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping


def train_and_evaluate_model():
    """
    Обучение и оценка модели нейронной сети для классификации данных Iris.

    Эта функция выполняет следующие шаги:
    1. Загружает данные Iris.
    2. Нормализует данные.
    3. Разделяет данные на обучающую и тестовую выборки.
    4. Создает и обучает модель нейронной сети с использованием ранней остановки.
    5. Визуализирует кривые обучения.
    6. Оценивает модель на тестовых данных.
    7. Выводит отчет о классификации, матрицу ошибок и ROC-кривую.
    """
    # Загрузка данных
    data = load_iris()
    X = data.data
    y = data.target

    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Создание модели нейронной сети
    model = Sequential()
    model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))  # 3 класса

    # Компиляция модели
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ранняя остановка
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Обучение модели
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # Визуализация кривых обучения
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Обучающая ошибка')
    plt.plot(history.history['val_loss'], label='Валидационная ошибка')
    plt.title('Ошибка')
    plt.xlabel('Эпохи')
    plt.ylabel('Ошибка')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Обучающая точность')
    plt.plot(history.history['val_accuracy'], label='Валидационная точность')
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    plt.show()

    # Оценка модели на тестовых данных
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Точность на тестовых данных: {accuracy:.4f}')

    # Предсказание на тестовых данных
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Отчет о классификации
    print("Отчет о классификации:")
    print(classification_report(y_test, y_pred_classes, target_names=data.target_names))

    # Матрица ошибок
    print("Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred_classes))

    # ROC Curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Построение ROC Curve
    plt.figure()
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f'ROC кривая класса {data.target_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')  # Случайный классификатор
    plt.xlabel('False Positive Rate (FPR) / Ложноположительная ставка')
    plt.ylabel('True Positive Rate (TPR) / Истинноположительная ставка')
    plt.title('ROC кривая')
    plt.legend(loc="lower right")
    plt.show()


# Запуск функции
train_and_evaluate_model()
