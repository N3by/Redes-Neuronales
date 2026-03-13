import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Crear modelo simple de decisión (4 entradas -> 3 opciones)
model = keras.Sequential([
    layers.Dense(3, input_shape=(4,), activation='softmax', use_bias=True)
])

# 2. Pesos manuales para:
# [hacer ejercicio en la mañana, hacer ejercicio en la tarde, no hacer ejercicio]
# Entradas: [energia, tiempo_libre, clima, motivacion]
pesos = np.array([
    [2.0, 0.8, -1.5],   # energia
    [0.7, 1.8, -1.0],   # tiempo_libre
    [1.4, 0.6, -1.6],   # clima
    [2.0, 1.4, -2.0]    # motivacion
])  # shape (4, 3)
sesgo = np.array([0.9, 0.1, 0.3])  # shape (3,)

# Asignar pesos y sesgo a la capa
model.layers[0].set_weights([pesos, sesgo])

# 3. Entrada de prueba (valores entre 0 y 1)
# energia, tiempo_libre, clima, motivacion
entrada = np.array([[0.8, 0.2, 0.9, 0.8]])

# 4. Predecir
salida = model.predict(entrada)

# 5. Mostrar resultado
opciones = [
    "Hacer ejercicio hoy en la mañana",
    "Hacer ejercicio hoy en la tarde",
    "No hacer ejercicio hoy"
]

indice = np.argmax(salida[0])
print("Probabilidades por opción:")
for i, opcion in enumerate(opciones):
    print(f"- {opcion}: {salida[0][i]:.4f}")

print(f"\nDecisión final: {opciones[indice]}")
