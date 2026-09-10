"""
Una red neuronal desde cero, sin librerías de deep learning. Solo numpy.

Aprende XOR:  0,0 -> 0    0,1 -> 1    1,0 -> 1    1,1 -> 0

XOR es el ejemplo clásico porque NO se puede resolver con una sola neurona:
no hay ninguna recta que separe los ceros de los unos. Hace falta una capa
oculta, y por eso este ejemplo mínimo ya contiene todo lo esencial.
"""

import numpy as np

np.random.seed(42)  # para que salga igual cada vez

# ---------------------------------------------------------------- datos
# 4 ejemplos, cada uno con 2 entradas y 1 salida esperada.
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)

y = np.array([[0],
              [1],
              [1],
              [0]], dtype=float)

# ---------------------------------------------------- activación y derivada
def sigmoid(x):
    """Aplasta cualquier número al rango (0, 1). Suave, no un umbral brusco."""
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(salida):
    """Pendiente de la sigmoide, expresada a partir de su propia salida."""
    return salida * (1 - salida)

# ------------------------------------------------------------ los pesos
# ESTO es lo único que la red aprende. Empiezan aleatorios: pura basura.
W1 = np.random.uniform(-1, 1, (2, 2))   # entradas -> capa oculta
b1 = np.zeros((1, 2))
W2 = np.random.uniform(-1, 1, (2, 1))   # capa oculta -> salida
b2 = np.zeros((1, 1))

def predecir(X):
    oculta = sigmoid(X @ W1 + b1)
    return sigmoid(oculta @ W2 + b2), oculta

print("=" * 62)
print("ANTES DE ENTRENAR (pesos aleatorios)")
print("=" * 62)
print("W1 =\n", np.round(W1, 3))
print("W2 =\n", np.round(W2, 3))
salida, _ = predecir(X)
for entrada, pred, esperado in zip(X, salida, y):
    print(f"  {entrada} -> {pred[0]:.3f}   (esperado {esperado[0]:.0f})")

# ------------------------------------------------------- bucle de aprendizaje
LR = 0.5          # learning rate: cuánto movemos los pesos en cada paso
EPOCAS = 20000

print()
print("=" * 62)
print("ENTRENANDO")
print("=" * 62)
print(f"{'época':>7}  {'error':>9}   predicciones (0,0) (0,1) (1,0) (1,1)")

for epoca in range(EPOCAS + 1):
    # 1. FORWARD: la red produce su respuesta
    oculta = sigmoid(X @ W1 + b1)
    salida = sigmoid(oculta @ W2 + b2)

    # 2. ERROR: cuánto se equivoca
    error = salida - y
    loss = np.mean(error ** 2)

    # 3. BACKWARD: cuánta culpa tiene cada peso del error (regla de la cadena)
    d_salida = error * sigmoid_deriv(salida)
    d_W2 = oculta.T @ d_salida
    d_b2 = d_salida.sum(axis=0, keepdims=True)

    d_oculta = (d_salida @ W2.T) * sigmoid_deriv(oculta)
    d_W1 = X.T @ d_oculta
    d_b1 = d_oculta.sum(axis=0, keepdims=True)

    # 4. ACTUALIZAR: mover cada peso en la dirección que reduce el error
    W2 -= LR * d_W2
    b2 -= LR * d_b2
    W1 -= LR * d_W1
    b1 -= LR * d_b1

    if epoca % 2000 == 0:
        preds = "  ".join(f"{v:.2f}" for v in salida.ravel())
        print(f"{epoca:>7}  {loss:>9.5f}   {preds}")

# ------------------------------------------------------------- resultado
print()
print("=" * 62)
print("DESPUÉS DE ENTRENAR")
print("=" * 62)
print("W1 =\n", np.round(W1, 3))
print("W2 =\n", np.round(W2, 3))
salida, _ = predecir(X)
for entrada, pred, esperado in zip(X, salida, y):
    ok = "OK" if round(pred[0]) == esperado[0] else "MAL"
    print(f"  {entrada} -> {pred[0]:.3f}   (esperado {esperado[0]:.0f})  {ok}")
