import numpy as np

n, m = 10, 9

# Генерация массивов A и B
A = np.random.randn(n, m)
B = np.random.randn(n, m)

# 1. Найти минимальное значение в A и максимальное значение в B
min_A = np.min(A)
max_B = np.max(B)

# 2. Найти сумму всех элементов в A
sum_A = np.sum(A)

# 3. Найти среднее значение всех элементов в B
mean_B = np.mean(B)

# 4. Умножить каждый элемент в A на среднее значение в B
A_multiplied_by_mean_B = A * mean_B

# 5. Найти сумму всех элементов в B по столбцам
sum_B_columns = np.sum(B, axis=0)

# 6. Найти максимальное значение в A по строкам
max_A_rows = np.max(A, axis=1)

# 7. Вычислить произведение A и B (поэлементное произведение)
A_B_product = np.multiply(A, B)

# 8. Сделать массивы A и B одномерными и объединить их в один
#  двумерный массив C размерностью (n+m, 2)
A_flat = A.flatten()
B_flat = B.flatten()
C = np.column_stack((A_flat, B_flat))

# Вывод результатов
print(f"Массив A {A}")
print(f"Массив B {B}")
print(f"Минимальное значение в A: {min_A}")
print(f"Максимальное значение в B: {max_B}")
print(f"Сумма всех элементов в A: {sum_A}")
print(f"Среднее значение всех элементов в B: {mean_B}")
print(f"Результат умножения каждого элемента в A на среднее значение B:\n{A_multiplied_by_mean_B}")
print(f"Сумма всех элементов в B по столбцам: {sum_B_columns}")
print(f"Максимальные значения в A по строкам: {max_A_rows}")
print(f"Произведение массивов A и B:\n{A_B_product}")
print(f"Размер объединенного массива C: {C.shape}")
