import lesson4module as lm
from sklearn.model_selection import train_test_split

openfile = r'C:\\Users\User\PycharmProjects\lesson4\data1.csv'
df = lm.load_data(openfile)

# Смотрим кол-во пропущенных значения в каждой колонке
lm.info_mis_val(df)

# Применяем метод для заполнения пустых колонок
df = lm.missing_values(df)

# Размечаем данные
target_column = 'USD'
X, y = lm.preprocess_data(df, target_column)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Обучение модели
model = lm.train_model(X_train, y_train)

# Предсказание на тестовых данных
y_pred = lm.predict(model, X_test)

# Метод построения гистограмм
lm.histogramma(y_test, y_pred)

# Метод построения диограмм распределений
lm.scatter_diagram(y_test, y_pred)

# Метод построения линейного графика
lm.line_graphs(y_test, y_pred)


