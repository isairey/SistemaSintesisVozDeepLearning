# 🎙️ SistemaSintesisVozDeepLearning - Text to Speech Synthesis

Sistema de síntesis de voz basado en **Tacotron 2**, desarrollado con **PyTorch** para convertir texto en voz natural mediante redes neuronales profundas. Este proyecto implementa la arquitectura propuesta en el artículo *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions*, utilizando espectrogramas Mel para generar audio de alta calidad.

---

## 📌 Características

* Conversión de texto a voz (TTS) basada en Deep Learning.
* Implementación desarrollada en PyTorch.
* Soporte para entrenamiento distribuido (Multi-GPU).
* Compatible con Automatic Mixed Precision (AMP).
* Utiliza el conjunto de datos LJSpeech.
* Generación de espectrogramas Mel.
* Compatible con modelos preentrenados.
* Preparado para integración con vocoders como WaveGlow.

---

## 🛠️ Tecnologías Utilizadas

* Python 3
* PyTorch
* CUDA
* cuDNN
* NVIDIA Apex
* TensorBoard
* Jupyter Notebook

---

## 📂 Estructura del Proyecto

```bash
SistemaSintesisVozDeepLearning/
├── data/
├── filelists/
├── hparams.py
├── train.py
├── inference.ipynb
├── requirements.txt
├── tensorboard.png
└── README.md
```

---

## ⚙️ Requisitos Previos

Antes de comenzar, asegúrate de contar con:

* GPU NVIDIA compatible con CUDA.
* CUDA Toolkit instalado.
* cuDNN configurado correctamente.
* Python 3.8 o superior.
* Git.

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/isairey/SistemaSintesisVozDeepLearning.git
cd SistemaSintesisVozDeepLearning
```

### 2. Inicializar submódulos

```bash
git submodule init
git submodule update
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Instalar PyTorch

```bash
pip install torch torchvision torchaudio
```

### 5. Instalar Apex

```bash
git clone https://github.com/isaireyOptimizacionDeepLearning
cd OptimizacionDeepLearning
pip install -v --disable-pip-version-check --no-cache-dir ./
```

---

## 📚 Preparación del Dataset

Descargar el dataset **LJSpeech**:

https://keithito.com/LJ-Speech-Dataset/

Extraer los archivos y actualizar las rutas en:

```bash
filelists/*.txt
```

Sustituyendo:

```text
DUMMY
```

por la ruta correspondiente a:

```text
ljs_dataset_folder/wavs
```

---

## 🎯 Entrenamiento

Para iniciar el entrenamiento:

```bash
python train.py --output_directory=outdir --log_directory=logdir
```

Visualizar métricas con TensorBoard:

```bash
tensorboard --logdir=outdir/logdir
```

---

## 🔥 Entrenamiento con Modelo Preentrenado

Descarga un modelo Tacotron 2 previamente entrenado y ejecuta:

```bash
python train.py \
--output_directory=outdir \
--log_directory=logdir \
-c tacotron2_statedict.pt \
--warm_start
```

Beneficios:

* Convergencia más rápida.
* Menor tiempo de entrenamiento.
* Mejor estabilidad inicial.

---

## ⚡ Entrenamiento Multi-GPU y AMP

Para utilizar múltiples GPUs y precisión mixta:

```bash
python -m multiproc train.py \
--output_directory=outdir \
--log_directory=logdir \
--hparams=distributed_run=True,fp16_run=True
```

Ventajas:

* Mayor velocidad de entrenamiento.
* Menor consumo de memoria GPU.
* Escalabilidad para datasets grandes.

---

## 🎧 Inferencia

### 1. Descargar modelos

* Tacotron 2
* WaveGlow

### 2. Ejecutar Jupyter Notebook

```bash
jupyter notebook --ip=127.0.0.1 --port=31337
```

### 3. Abrir

```text
inference.ipynb
```

Introducir texto y generar audio sintetizado.

---

## 📊 Resultados

El sistema genera:

* Alineaciones de atención.
* Espectrogramas Mel predichos.
* Espectrogramas Mel reales.
* Audio sintetizado de alta calidad.

Además, puede monitorearse el proceso de entrenamiento mediante TensorBoard.

---

## 🔗 Modelos Relacionados

### WaveGlow

Vocoder basado en Flow Networks para generación de audio en tiempo real.

### nv-WaveNet

Implementación optimizada de WaveNet para síntesis de voz de alta velocidad.

---

## 📖 Referencias

* Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions.
* LJSpeech Dataset.
* PyTorch.
* NVIDIA Apex.
* WaveGlow.
* Tacotron 2.

---

## 👨‍💻 Desarrollador

Isai Reyes - FullStack Developer

---

## 📜 Licencia

MIT
