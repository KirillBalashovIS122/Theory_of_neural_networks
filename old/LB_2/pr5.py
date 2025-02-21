import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Исходная таблица:")
    print(df)
    return df

def sort_by_wednesday_temperature(df):
    df_sorted = df.sort_values(by='СР', ascending=False)
    print("\nОтсортированная таблица по столбцу 'СР' (в порядке убывания):")
    print(df_sorted)
    return df_sorted

def filter_friday_above_average(df):
    average_friday = df['ПТ'].mean()
    df_friday_above_avg = df[df['ПТ'] > average_friday]
    print("\nСтроки, где значение в 'ПТ' больше среднего по столбцу:")
    print(df_friday_above_avg)
    return df_friday_above_avg

def calculate_average_temperature(df):
    df['Средняя температура'] = df.iloc[:, 1:].mean(axis=1)
    print("\nСредняя температура по каждому городу:")
    print(df[['Город', 'Средняя температура']])
    return df

def replace_below_average_tuesday(df):
    average_tuesday = df['ВТ'].mean()
    df['ВТ'] = df['ВТ'].apply(lambda x: average_tuesday if x < average_tuesday else x)
    print("\nТаблица после замены значений в 'ВТ', которые меньше среднего:")
    print(df)
    return df

def add_weekly_average_column(df):
    df['Средняя за неделю'] = df.iloc[:, 1:-1].mean(axis=1)
    print("\nТаблица с добавленным столбцом 'Средняя за неделю':")
    print(df)
    return df

def save_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"\nОбработанные данные сохранены по пути: {file_path}")

def main():
    input_file = '/home/kbalashov/VS_Code/TONN/LB_2/weather_forecast.csv'
    df = load_data(input_file)
    df_sorted = sort_by_wednesday_temperature(df)
    df_friday_above_avg = filter_friday_above_average(df)
    df = calculate_average_temperature(df)
    df = replace_below_average_tuesday(df)
    df = add_weekly_average_column(df)
    output_file = '/home/kbalashov/VS_Code/TONN/LB_2/processed_weather_forecast.csv'
    save_data(df, output_file)

if __name__ == "__main__":
    main()
