import h2o
from h2o.automl import H2OAutoML
from sklearn.datasets import load_iris
import pandas as pd

# Инициализация H2O с явным указанием памяти
h2o.init(nthreads=-1, max_mem_size="2G")  # Важно указать достаточную память

# Загрузка данных через pandas для корректной конвертации
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = pd.Series(iris.target).astype('category')  # Правильное преобразование категорий

# Разделение данных
train, test = h2o.H2OFrame(df).split_frame(ratios=[0.8], seed=42)

# Указание целевой переменной и предикторов
predictors = iris.feature_names
target = 'target'

# Конфигурация AutoML
aml = H2OAutoML(
    max_models=10,
    seed=42,
    max_runtime_secs=300,  # Увеличиваем время выполнения
    exclude_algos=["StackedEnsemble"]  # Иногда помогает отключение ансамблей
)

# Запуск обучения
aml.train(x=predictors, y=target, training_frame=train)

# Проверка результатов
print(aml.leaderboard)