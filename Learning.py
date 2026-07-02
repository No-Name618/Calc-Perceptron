import math
import random

# ------------------- Функции активации -------------------
def sigmoid(x):
    # Защита от переполнения math.exp
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

# ------------------- Генерация данных -------------------
def generate_dataset(n=1000, bias_right=0.5):
    """
    Генерирует n точек. 
    Границы сужены, так как лемниската теперь вписана в [-1, 1].
    """
    data = []
    for _ in range(n):
        if random.random() < bias_right:
            x = random.uniform(0, 2)      # правая половина
        else:
            x = random.uniform(-2, 0)     # левая половина
        y = random.uniform(-1.5, 1.5)
        
        # Убрали коэффициент 2 перед второй скобкой
        val = (x**2 + y**2)**2 - 1.0 * (x**2 - y**2)
        label = 1 if val <= 0 else 0
        data.append((x, y, label))
    return data

def prepare_data(data):
    """Оставляет данные без изменений (нормализация убрана)."""
    return data

# ------------------- Инициализация весов (5 нейронов в скрытом слое) -------------------
def init_weights():
    # Скрытый слой: 5 нейронов, каждый с 2 входами + смещение
    # Используем случайные веса, так как старые были подобраны для нормализованных данных
    w = [[random.uniform(-1.0, 1.0) for _ in range(2)] for _ in range(5)]
    b = [random.uniform(-1.0, 1.0) for _ in range(5)]
    
    # Выходной слой: 5 весов + смещение
    v = [random.uniform(-1.0, 1.0) for _ in range(5)]
    b_out = random.uniform(-1.0, 1.0)
    
    return w, b, v, b_out

# ------------------- Обучение -------------------
def train(data, w, b, v, b_out, lr=0.01, epochs=1000):
    n = len(data)
    hidden_size = len(w)  # должно быть 5
    
    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0.0
        
        for x1, x2, y_true in data:
            # ----- Прямой проход -----
            # Скрытый слой
            z = [0.0] * hidden_size
            a = [0.0] * hidden_size
            for i in range(hidden_size):
                z[i] = w[i][0] * x1 + w[i][1] * x2 + b[i]
                a[i] = sigmoid(z[i])

            # Выходной слой
            z_out = sum(v[i] * a[i] for i in range(hidden_size)) + b_out
            a_out = sigmoid(z_out)
            
            # Потеря (BCE)
            loss = - (y_true * math.log(a_out + 1e-15) +
                      (1 - y_true) * math.log(1 - a_out + 1e-15))
            total_loss += loss
            
            # ----- Обратное распространение -----
            delta_out = a_out - y_true
            
            # Градиенты для скрытого слоя
            delta_h = [0.0] * hidden_size
            for i in range(hidden_size):
                delta_h[i] = delta_out * v[i] * a[i] * (1 - a[i])
                
            # Обновление выходного слоя
            for i in range(hidden_size):
                v[i] -= lr * delta_out * a[i]
            b_out -= lr * delta_out
            
            # Обновление скрытого слоя
            for i in range(hidden_size):
                w[i][0] -= lr * delta_h[i] * x1
                w[i][1] -= lr * delta_h[i] * x2
                b[i] -= lr * delta_h[i]
                
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, loss = {total_loss/n:.6f}")

# ------------------- Предсказание и точность -------------------
def predict(x, y, w, b, v, b_out):
    """Предсказание для исходных координат (без нормализации)."""
    hidden_size = len(w)
    # Скрытый слой
    a = [0.0] * hidden_size
    for i in range(hidden_size):
        z = w[i][0] * x + w[i][1] * y + b[i]
        a[i] = sigmoid(z)
        
    # Выходной слой
    z_out = sum(v[i] * a[i] for i in range(hidden_size)) + b_out
    return sigmoid(z_out)

def accuracy(data, w, b, v, b_out):
    correct = 0
    for x, y, label in data:
        pred = predict(x, y, w, b, v, b_out)
        if (pred >= 0.5) == label:
            correct += 1
    return correct / len(data)

# ------------------- Визуализация (консоль) -------------------
def visualize(w, b, v, b_out, step=0.2):
    """Рисует карту классификации в квадрате [-2, 2]."""
    print("\nКарта классификации (# = класс 1, . = класс 0): ")
    y_vals = [round(y, 2) for y in [i * step for i in range(-10, 11)]]
    x_vals = [round(x, 2) for x in [i * step for i in range(-10, 11)]]
    
    for y in reversed(y_vals):
        line = ""
        for x in x_vals:
            p = predict(x, y, w, b, v, b_out)
            line += "# " if p >= 0.5 else ". "
        print(line)

# ------------------- Главная функция -------------------
if __name__ == "__main__":
    # Генерация данных (увеличим выборку для стабильности без нормализации)
    train_data = generate_dataset(n=5000)
    test_data  = generate_dataset(n=500, bias_right=0.5)
    
    train_prepared = prepare_data(train_data)
    test_prepared  = prepare_data(test_data)
    
    # Инициализация весов (5 нейронов)
    w, b, v, b_out = init_weights()
    
    # Обучение
    print("Обучение...")
    train(train_prepared, w, b, v, b_out, lr=0.01, epochs=10000)
    
    # Точность
    acc_train = accuracy(train_data, w, b, v, b_out)
    acc_test  = accuracy(test_data,  w, b, v, b_out)
    print(f"\nТочность на обучении: {acc_train*100:.2f}%")
    print(f"Точность на тесте:    {acc_test*100:.2f}%")
    
    # Визуализация
    visualize(w, b, v, b_out)
    
    # Вывод выражения для Casio fx-82ES Plus
    print("\n" + "="*60)
    print("ВЫРАЖЕНИЕ ДЛЯ ИНФЕРЕНСА НА КАЛЬКУЛЯТОРЕ (5 нейронов)")
    print("="*60)
    print("Для исходных координат x и y (из диапазона [-2, 2]):")
    print("1. Подставьте x и y напрямую в формулы:")
    
    for i in range(len(w)):
        print(f"   a_{i+1} = 1/(1+exp(-({w[i][0]:.6f}*x + {w[i][1]:.6f}*y + {b[i]:.6f})))")
        
    # Выход
    expr = " + ".join(f"{v[i]:.6f}*a_{i+1}" for i in range(len(v)))
    print(f"   z = 1/(1+exp(-({expr} + {b_out:.6f})))")
    print("2. Если z >= 0.5, класс 1 (внутри восьмёрки), иначе класс 0.")
    print("\nПримечание: на калькуляторе e^x заменяется на EXP, а скобки расставляются вручную.")
