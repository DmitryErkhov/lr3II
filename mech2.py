#2
import numpy as np

# Определение треугольной функции принадлежности
def triangular_membership(x, a, b, c):
    """
    Вычисляет принадлежность x для треугольной функции с параметрами a, b, c
    """
    if x <= a or x >= c:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    else:
        return 0

# Функции принадлежности для здоровья (входная переменная)
def health_membership(x_values):
    """
    Возвращает функции принадлежности для Low, Medium, High для заданного массива x_values
    """
    mu_low = np.array([triangular_membership(x, 1, 15, 40) for x in x_values])
    mu_medium = np.array([triangular_membership(x, 31, 45, 70) for x in x_values])
    mu_high = np.array([triangular_membership(x, 61, 80, 100) for x in x_values])
    return mu_low, mu_medium, mu_high

# Функции принадлежности для рекомендации (выходная переменная)
def recommendation_membership(y_values):
    """
    Возвращает функции принадлежности для Heal, Avoid, Attack для заданного массива y_values
    """
    mu_heal = np.array([triangular_membership(y, 0, 10, 40) for y in y_values])
    mu_avoid = np.array([triangular_membership(y, 30, 50, 70) for y in y_values])
    mu_attack = np.array([triangular_membership(y, 60, 90, 100) for y in y_values])
    return mu_heal, mu_avoid, mu_attack

# Построение матрицы R для каждого правила
def calculate_R(mu_A, mu_B):
    """
    Вычисляет матрицу R (4 строки x 3 столбца)
    """
    R = np.zeros((len(mu_A), len(mu_B)))
    for i, mu_x in enumerate(mu_A):
        for j, mu_y in enumerate(mu_B):
            R[i, j] = min(mu_x, mu_y)
    return R

# Построение агрегированной матрицы R'
def aggregate_rules(R_list):
    """
    Агрегирует правила через максимум
    """
    return np.maximum.reduce(R_list)

# Применение композиции
def calculate_B_aggregate(R_aggregate):
    """
    Применяет композицию (A' ◦ R') для агрегированной матрицы
    """
    return np.max(R_aggregate, axis=0)

# Дефазификация методом центра тяжести
def defuzzify_center_of_gravity(y, mu_y):
    return np.sum(y * mu_y) / np.sum(mu_y)

# Задаём значение здоровья
x_health = 30  # Пример текущего значения здоровья

# Определяем множества X и Y
x_values = np.array([10, 30, 50, 70])  # Множество X (4 уровня здоровья)
y_values = np.array([10, 50, 90])  # Множество Y (3 точки: Heal, Avoid, Attack)

# Вычисляем функции принадлежности для X
mu_low, mu_medium, mu_high = health_membership(x_values)

# Вычисляем степени принадлежности для конкретного здоровья
health_values = health_membership([x_health])

# Вычисляем функции принадлежности для Y
mu_heal, mu_avoid, mu_attack = recommendation_membership(y_values)

# Матрицы R для правил
R1 = calculate_R(mu_low, mu_heal)  # Low -> Heal
R2 = calculate_R(mu_medium, mu_avoid)  # Medium -> Avoid
R3 = calculate_R(mu_high, mu_attack)  # High -> Attack

# Агрегированная матрица R'
R_aggregate = aggregate_rules([R1, R2, R3])

# Итоговый нечёткий выход B'(y)
B_aggregate = calculate_B_aggregate(R_aggregate)

# Дефазификация
crisp_output = defuzzify_center_of_gravity(y_values, B_aggregate)

# Вывод результатов
print("\nТекущее значение здоровья:", x_health)

print("\nСтепени принадлежности для текущего здоровья:")
print({
    "Low": health_values[0][0],
    "Medium": health_values[1][0],
    "High": health_values[2][0]
})

print("\nМатрица R1 (Low -> Heal):")
print(R1)

print("\nМатрица R2 (Medium -> Avoid):")
print(R2)

print("\nМатрица R3 (High -> Attack):")
print(R3)

print("\nАгрегированная матрица R':")
print(R_aggregate)

print("\nИтоговый выход B'(y):", B_aggregate)

print("\nЧёткий вывод (рекомендация):", crisp_output)