import matplotlib.pyplot as plt

# Данные по дням недели и температурам для города Москва
days = ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье']
temp_morning = [1, 3, 5, 4, 2, -1, 0]  # Температура утром
temp_day = [4, 5, 6, 6, 5, 6, 4]  # Температура днем
temp_evening = [0, -2, 1, 3, 3, 4, 4]  # Температура вечером

fig, ax = plt.subplots(3, 1, figsize=(10, 8))

# График 1 - температура утром
ax[0].plot(days, temp_morning, marker='o', color='blue', linestyle='--', label='Утром')
ax[0].set_title('Температурные изменения в Москве утром')
ax[0].set_xlabel('Дни недели')
ax[0].set_ylabel('Температура (°C)')
ax[0].legend()
ax[0].grid(True)

# График 2 - температура днем
ax[1].plot(days, temp_day, marker='o', color='green', linestyle='-', label='Днем')
ax[1].set_title('Температурные изменения в Москве днем')
ax[1].set_xlabel('Дни недели')
ax[1].set_ylabel('Температура (°C)')
ax[1].legend()
ax[1].grid(True)

# График 3 - температура вечером
ax[2].plot(days, temp_evening, marker='o', color='red', linestyle='-.', label='Вечером')
ax[2].set_title('Температурные изменения в Москве вечером')
ax[2].set_xlabel('Дни недели')
ax[2].set_ylabel('Температура (°C)')
ax[2].legend()
ax[2].grid(True)

# Настройка макета и сохранение графика
plt.tight_layout()
plt.savefig('temperature_moscow.png')
plt.show()
