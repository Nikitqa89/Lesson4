import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Метод для загрузки данных
def load_data(openfile):
    return pd.read_csv(openfile, delimiter=';', decimal=',')

# Метод для подсчета пропущенных значения в каждой колонке
def info_mis_val(df):
    print(df.isnull().sum())

# Метод для заполнения пустых значений. В нашем случае нужны значения на предыдущую дату
def missing_values(df):
    for column in df.columns:
        if df[column].isnull().sum() > 0:
            df[column] = df[column].ffill()
    return df

# Предобработка
def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column, 'date'])
    y = df[target_column]
    return X, y

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict(model, X):
    return model.predict(X)

# Построение гистограмм
def histogramma(y_test, y_pred):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(y_test, bins=33, alpha=0.5, color='green')
    ax[0].set_title('Реальный курс')
    ax[1].hist(y_pred, bins=33, alpha=0.5, color='red')
    ax[1].set_title('Предсказанный курс')
    plt.show()

# Построение диаграмм рассеяния
def scatter_diagram(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, color='green', label='Истинные значения')
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Стоимость доллара США')
    plt.title(f'Истинные и предсказанные значения')
    plt.legend()
    plt.show()

# Построение линейных графиков
def line_graphs(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, color='green', label='Истинные значения')
    plt.plot(range(len(y_pred)), y_pred, color='red', label='Предсказанные значения')
    plt.xlabel('Индекс')
    plt.ylabel('Стоимость доллара США')
    plt.title(f'Истинные и предсказанные значения')
    plt.legend()
    plt.show()


