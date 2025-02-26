# principalcode.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input  # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore

def entrenar_red_neuronal():
    # Cargar los datos
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()

    # Mostrar información sobre los datos
    print("Forma de los datos de entrenamiento:", train_data_x.shape)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[1])
    print("Forma de los datos de prueba:", test_data_x.shape)
    plt.imshow(train_data_x[1])
    plt.title("Ejemplo de imagen de entrenamiento")
    plt.show()

    # Definir la arquitectura de la red neuronal
    model = Sequential([
        Input(shape=(28 * 28,)),
        Dense(512, activation='relu'),  # Capa oculta con 512 neuronas
        Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada clase)
    ])

    # Compilar el modelo
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',  # Función de pérdida
        metrics=['accuracy']
    )

    # Resumen del modelo
    print("Resumen del modelo:")
    model.summary()

    # Preprocesamiento de los datos de entrenamiento
    x_train = train_data_x.reshape(60000, 28 * 28)
    x_train = x_train.astype('float32') / 255  # Normalización
    y_train = to_categorical(train_labels_y)

    # Preprocesamiento de los datos de prueba
    x_test = test_data_x.reshape(10000, 28 * 28)
    x_test = x_test.astype('float32') / 255  # Normalización
    y_test = to_categorical(test_labels_y)

    # Entrenar el modelo
    print("Entrenando la red neuronal...")
    model.fit(x_train, y_train, epochs=10, batch_size=128)

    # Evaluar el modelo
    print("Evaluando la red neuronal...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Pérdida: {loss}, Precisión: {accuracy}")

    # Mostrar gráficos
    plt.show()