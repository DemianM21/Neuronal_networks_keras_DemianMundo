# 🌟 Proyecto de Red Neuronal para Clasificación de Dígitos MNIST

Este proyecto implementa una red neuronal utilizando **TensorFlow** y **Keras** para clasificar dígitos escritos a mano del conjunto de datos MNIST.

📌 **Tecnologías utilizadas:** Python, TensorFlow, Keras, NumPy, Matplotlib

---

## 📂 Estructura del Proyecto

📁 **Organización del código:**
```bash
proyecto/
├── src/
│   ├── principalcode.py       # Lógica de la red neuronal
├── main.py                    # Punto de entrada del programa
├── Readme.md                  # Documentación del proyecto
├── requirements.txt            # Dependencias del proyecto
```

---

## 🔧 Requisitos

Para ejecutar este proyecto, necesitas:

✅ **Python 3.10** (por compatibilidad con TensorFlow).
✅ Las siguientes bibliotecas de Python (listadas en `requirements.txt`):
- `numpy`
- `matplotlib`
- `tensorflow`

---

## 🚀 Instalación

### 1️⃣ Clona este repositorio:
```bash
git clone https://github.com/DemianM21/Neuronal_networks_keras_DemianMundo.git
cd Neuronal_networks_keras_DemianMundo
```

### 2️⃣ Crea un entorno virtual (opcional pero recomendado):
```bash
python -m venv env
```

### 3️⃣ Activa el entorno virtual:
**En Windows:**
```bash
env\Scripts\activate
```
**En macOS/Linux:**
```bash
source env/bin/activate
```

### 4️⃣ Instala las dependencias:
```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución

Para ejecutar el proyecto, sigue estos pasos:
```bash
python main.py
```

---

## 📌 ¿Qué hace el proyecto?

🔹 **Carga el conjunto de datos MNIST:**
El conjunto de datos **MNIST** contiene imágenes de dígitos escritos a mano (0-9) y sus etiquetas correspondientes.

🔹 **Preprocesa los datos:**
- Normaliza las imágenes (valores entre 0 y 1).
- Redimensiona las imágenes para ser procesadas por la red neuronal.

🔹 **Define la arquitectura de la red neuronal:**
- **Capa de entrada:** 784 neuronas (28x28 píxeles).
- **Capa oculta:** 512 neuronas con función de activación **ReLU**.
- **Capa de salida:** 10 neuronas (una por cada dígito) con función de activación **softmax**.

🔹 **Entrena la red neuronal:**
- Se entrena durante **10 épocas** con un tamaño de lote (**batch size**) de **128**.

🔹 **Evalúa el modelo:**
- Se mide la precisión del modelo en el conjunto de prueba.

🔹 **Muestra gráficos:**
- Se visualiza un ejemplo de imagen del conjunto de datos de entrenamiento.

---

## 📊 Resultados esperados

📌 **Al ejecutar el programa, verás:**

✅ **Información sobre los datos:**
- Forma de los datos de entrenamiento y prueba.
- Etiqueta del primer ejemplo de entrenamiento.

✅ **Resumen del modelo:**
- Descripción de las capas y parámetros de la red neuronal.

✅ **Progreso del entrenamiento:**
- Pérdida y precisión en cada época.

✅ **Evaluación final:**
- Pérdida y precisión en el conjunto de prueba.

✅ **Gráficos:**
- Una imagen de ejemplo del conjunto de entrenamiento.

---
# Entrenamiento de una Red Neuronal en MNIST

El codigo que el proyecto implementa para construir una red neuronal en keras/Tenserflow para clasificar imagenes del dataset MNIST, es el siguiente:

## 1. Importación de Librerías
Se importan las librerías necesarias para manipular datos, visualizar imágenes y construir la red neuronal.

```python
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Input  # type: ignore
from keras.utils import to_categorical  # type: ignore
from keras.datasets import mnist  # type: ignore
```

## 2. Definición de la Función Principal
Se define la función `entrenar_red_neuronal()` que entrena y evalúa la red neuronal.

```python
def entrenar_red_neuronal():
    """
    Entrena una red neuronal para clasificar imágenes del dataset MNIST.
    """
```

## 3. Carga de Datos
Se carga el dataset MNIST, dividiéndolo en datos de entrenamiento y prueba.

```python
    (train_data_x, train_labels_y), (test_data_x, test_labels_y) = mnist.load_data()
```

## 4. Información y Visualización de Datos
Se muestran detalles del conjunto de datos y un ejemplo de imagen de entrenamiento.

```python
    print("Forma de los datos de entrenamiento:", train_data_x.shape)
    print("Etiqueta del primer ejemplo de entrenamiento:", train_labels_y[1])
    print("Forma de los datos de prueba:", test_data_x.shape)
    
    plt.imshow(train_data_x[1])
    plt.title("Ejemplo de imagen de entrenamiento")
    plt.show()
```

## 5. Definición de la Arquitectura del Modelo
Se crea un modelo secuencial con una capa de entrada, una capa oculta con 512 neuronas y una capa de salida con 10 neuronas.

```python
    model = Sequential([
        Input(shape=(28 * 28,)),  # Capa de entrada con 784 nodos (28x28 píxeles aplanados)
        Dense(512, activation='relu'),  # Capa oculta con 512 neuronas y activación ReLU
        Dense(10, activation='softmax')  # Capa de salida con 10 neuronas (una por cada clase)
    ])
```

## 6. Compilación del Modelo
Se configura el modelo con la función de pérdida `categorical_crossentropy`, el optimizador `rmsprop` y la métrica de precisión.

```python
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',  # Función de pérdida
        metrics=['accuracy']
    )
    
    print("Resumen del modelo:")
    model.summary()
```

## 7. Preprocesamiento de Datos
Se normalizan las imágenes y se convierten las etiquetas a formato one-hot.

```python
    x_train = train_data_x.reshape(60000, 28 * 28)
    x_train = x_train.astype('float32') / 255
    y_train = to_categorical(train_labels_y)
    
    x_test = test_data_x.reshape(10000, 28 * 28)
    x_test = x_test.astype('float32') / 255
    y_test = to_categorical(test_labels_y)
```

## 8. Entrenamiento del Modelo
El modelo se entrena con 10 épocas y un tamaño de lote de 128.

```python
    print("Entrenando la red neuronal...")
    model.fit(x_train, y_train, epochs=10, batch_size=128)
```

## 9. Evaluación del Modelo
Se mide la pérdida y precisión del modelo en los datos de prueba.

```python
    print("Evaluando la red neuronal...")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Pérdida: {loss}, Precisión: {accuracy}")
```

## 10. Visualización
Se muestra la imagen de ejemplo y gráficos adicionales si fueran generados.

```python
    plt.show()
```


---

## 📢 Contacto y Créditos
📌 **Autor:** *Demian Mundo*  
📌 **Repositorio:** [GitHub](https://github.com/DemianM21/Neuronal_networks_keras_DemianMundo.git)

🚀 **¡Espero que disfrutes este proyecto! No dudes en contribuir o reportar problemas.**

