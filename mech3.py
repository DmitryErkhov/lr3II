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

# Вычисление уровня истинности предпосылок
def calculate_alpha(mu_low, mu_medium, mu_high):
    """
    Вычисляет уровни истинности предпосылок для каждого правила
    """
    alpha_1 = np.max(mu_low).item()  # Для правила Low -> Heal
    alpha_2 = np.max(mu_medium).item()  # Для правила Medium -> Avoid
    alpha_3 = np.max(mu_high).item()  # Для правила High -> Attack
    return alpha_1, alpha_2, alpha_3

# Определение выходного множества для каждого правила
def calculate_rule_output(alpha, mu_B):
    """
    Вычисляет выходное множество для правила: mu_B'(y) = min(alpha, mu_B(y))
    """
    return np.minimum(alpha, mu_B)

# Агрегирование выходов всех правил
def aggregate_outputs(B_prime_list):
    """
    Агрегирует выходы всех правил: B'(y) = max(B_1'(y), B_2'(y), ..., B_n'(y))
    """
    return np.maximum.reduce(B_prime_list)

# Дефазификация методом центра тяжести
def defuzzify_center_of_gravity(y, mu_y):
    return float(np.sum(y * mu_y) / np.sum(mu_y))

# Задаём значение здоровья
x_health = 30  # Пример текущего значения здоровья

# Определяем множества X и Y
x_values = np.array([10, 30, 50, 70])  # Множество X (4 уровня здоровья)
y_values = np.array([10, 50, 90])  # Множество Y (3 точки: Heal, Avoid, Attack)

# Вычисляем функции принадлежности для X
mu_low, mu_medium, mu_high = health_membership(x_values)

# Вычисляем функции принадлежности для конкретного значения x_health
health_values = health_membership([x_health])

# Вычисляем функции принадлежности для Y
mu_heal, mu_avoid, mu_attack = recommendation_membership(y_values)

# Вычисление уровней истинности предпосылок (альфа)
alpha_1, alpha_2, alpha_3 = calculate_alpha(mu_low, mu_medium, mu_high)

# Вычисление выходных множеств для каждого правила
B1_prime = calculate_rule_output(alpha_1, mu_heal)  # Heal
B2_prime = calculate_rule_output(alpha_2, mu_avoid)  # Avoid
B3_prime = calculate_rule_output(alpha_3, mu_attack)  # Attack

# Агрегированный выход
B_aggregate = aggregate_outputs([B1_prime, B2_prime, B3_prime])

# Дефазификация
crisp_output = defuzzify_center_of_gravity(y_values, B_aggregate)

# Вывод результатов
print("\nТекущее значение здоровья:", x_health)

print("\nСтепени принадлежности для текущего здоровья:")
print({
    "Low": float(health_values[0][0]),
    "Medium": float(health_values[1][0]),
    "High": float(health_values[2][0])
})

print("\nУровни истинности предпосылок:")
print({"Alpha 1 (Low -> Heal)": alpha_1, "Alpha 2 (Medium -> Avoid)": alpha_2, "Alpha 3 (High -> Attack)": alpha_3})

print("\nВыходное множество B1' (Heal):", B1_prime.tolist())
print("Выходное множество B2' (Avoid):", B2_prime.tolist())
print("Выходное множество B3' (Attack):", B3_prime.tolist())

print("\nАгрегированный выход B'(y):", B_aggregate.tolist())

print("\nЧёткий вывод (рекомендация):", crisp_output)
