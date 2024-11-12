import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение данных из файла CSV по указанному пути
data = pd.read_csv('/home/kbalashov/advertising.csv')

# Проверка данных
print(data.head())
print(data.info())

# Остальной код анализа, обучения и оценки модели
# Анализ корреляции
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.savefig('/home/kbalashov/VS_Code/TONN/LB_2/correlation_heatmap.png')
0# Сохраняем график корреляции
plt.close()  # Закрываем график после сохранения

# Подготовка данных
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R^2): {r2}')

# Визуализация результатов
plt.scatter(y_test, y_pred)
plt.xlabel("Фактические значения")
plt.ylabel("Предсказанные значения")
plt.title("Сравнение фактических и предсказанных продаж")

# Сохраняем график предсказаний
plt.savefig('/home/kbalashov/VS_Code/TONN/LB_2/predictions_plot.png')
plt.close()  # Закрываем график после сохранения
