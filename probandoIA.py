import tensorflow as tf
import numpy as np

fahrenheit = np.array([32, 33.8, 50, 69, 57, 25, 0], dtype=float)
celsius = np.array([0, 1, 10, 20.555, 13.888, -3.888, -17.777], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Comenzando entrenamiento!!!")
entrenamiento = modelo.fit(fahrenheit, celsius, epochs=500,verbose=False)
print("Listo...")
