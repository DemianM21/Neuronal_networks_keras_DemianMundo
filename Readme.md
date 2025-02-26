# Proyecto de Red Neuronal para Clasificación de Dígitos MNIST

Este proyecto implementa una red neuronal utilizando **TensorFlow** y **Keras** para clasificar dígitos escritos a mano del conjunto de datos MNIST.
El código está organizado en dos archivos principales:

- **principalcode.py**: Contiene la lógica de la red neuronal.
- **main.py**: Sirve como punto de entrada del programa.

## 📂 Estructura del Proyecto

```
proyecto/
├── src/
│   ├── principalcode.py       # Lógica de la red neuronal
├── main.py                    # Punto de entrada del programa
├── Readme.md                  # Documentación del proyecto
├── requirements.txt            # Dependencias del proyecto
```

## 🔧 Requisitos

Para ejecutar este proyecto, necesitas:

- **Python 3.10** (por compatibilidad con TensorFlow).
- Las siguientes bibliotecas de Python (listadas en `requirements.txt`):
  - `numpy`
  - `matplotlib`
  - `tensorflow`

## 🚀 Instalación

### 1️⃣ Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio
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

## ▶️ Ejecución

Para ejecutar el proyecto, sigue estos pasos:

```bash
python main.py
```

## 📌 ¿Qué hace el proyecto?

### 🔹 Carga el conjunto de datos MNIST
El conjunto de datos **MNIST** contiene imágenes de dígitos escritos a mano (0-9) y sus etiquetas correspondientes.

### 🔹 Preprocesa los datos
- Las imágenes se **normalizan** (valores entre 0 y 1).
- Se **redimensionan** para ser procesadas por la red neuronal.

### 🔹 Define la arquitectura de la red neuronal
La red neuronal cuenta con:
- **Capa de entrada**: 784 neuronas (28x28 píxeles).
- **Capa oculta**: 512 neuronas con función de activación **ReLU**.
- **Capa de salida**: 10 neuronas (una por cada dígito) con función de activación **softmax**.

### 🔹 Entrena la red neuronal
- Se entrena durante **10 épocas** con un tamaño de lote (**batch size**) de **128**.

### 🔹 Evalúa el modelo
- Se mide la precisión del modelo en el conjunto de prueba.

### 🔹 Muestra gráficos
- Se visualiza un ejemplo de imagen del conjunto de datos de entrenamiento.

## 📊 Resultados esperados
Al ejecutar el programa, verás:

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

📌 **Autor:** *Demian Mundo*  
📌 **Repositorio:** [GitHub](https://github.com/tu-usuario/tu-repositorio)

