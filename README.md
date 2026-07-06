# 📝 Handwritten Alphabet Recognition using CNN

Recognize handwritten English alphabets (A–Z) using a Convolutional Neural Network (CNN) trained on the EMNIST Letters dataset. This project leverages TensorFlow/Keras for deep learning and OpenCV for image preprocessing to accurately classify handwritten characters while displaying prediction confidence and the top three probable outputs.

![Python](https://img.shields.io/badge/python-3.10+-brightgreen)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Overview

Handwritten Alphabet Recognition is a deep learning project that classifies handwritten English alphabets using a CNN model trained on the EMNIST Letters dataset. The application preprocesses custom handwritten images, predicts the corresponding alphabet, and displays confidence scores along with the top three predictions.

### ✨ Features

- 🔤 Recognizes handwritten English alphabets (A–Z)
- 🧠 CNN model built using TensorFlow/Keras
- 🖼️ Image preprocessing using OpenCV
- 📊 Displays prediction confidence
- 🥇 Shows Top-3 predicted letters with probabilities
- ⚡ Fast and efficient inference
- 📈 Trained on the EMNIST Letters dataset

---

## 🛠️ Tech Stack

### Machine Learning

- TensorFlow
- Keras
- EMNIST Letters Dataset

### Image Processing

- OpenCV
- NumPy

### Visualization

- Matplotlib

### Programming Language

- Python

---

## 📁 Project Structure

```
HANDWRITTEN-ALPHABET-PREDICTION/
│
├── emnist_images/
│   ├── A_32.png
│   ├── D_55.png
│   └── ...
│
├── emnist_cnn_model.keras
├── train_model.py
├── predictions.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The model is trained on the **EMNIST Letters** dataset.

**Dataset Information**

- Classes: 26 (A–Z)
- Image Size: 28 × 28 pixels
- Grayscale images
- Thousands of handwritten samples

Dataset Link:

https://www.tensorflow.org/datasets/catalog/emnist

---

## 🧠 Model Architecture

The CNN consists of:

- Input Layer (28×28×1)
- Convolutional Layers
- Batch Normalization
- Max Pooling Layers
- Dropout Layers
- Fully Connected Dense Layer
- Softmax Output Layer (26 Classes)

---

## 🖼️ Image Preprocessing

Before prediction, every image undergoes:

- Grayscale conversion
- Background inversion (if required)
- Binary thresholding
- Contour detection
- Character cropping
- Padding
- Resize to 28×28
- Pixel normalization

These preprocessing steps improve prediction accuracy on custom handwritten inputs.

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Dataset | EMNIST Letters |
| Classes | 26 |
| Accuracy | ~96% |

The application predicts:

- Predicted Alphabet
- Confidence Score
- Top-3 Predictions

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- pip

### Step 1: Clone Repository

```bash
git clone https://github.com/<your-username>/handwritten-alphabet-prediction.git

cd handwritten-alphabet-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎮 Usage

### Train the Model

```bash
python train_model.py
```

This creates the trained model:

```
emnist_cnn_model.keras
```

### Predict a Handwritten Alphabet

1. Place your handwritten image inside the `emnist_images` folder.
2. Update the image filename in `predictions.py`.
3. Run:

```bash
python predictions.py
```

The program displays:

- Processed Input Image
- Predicted Alphabet
- Confidence Score
- Top-3 Predictions

---

## 📸 Sample Output

Example:

```
Prediction: D

Confidence: 99.82%

Top 3 Predictions

D : 99.82%
O : 0.11%
P : 0.07%
```

---

## 💡 Future Improvements

- 🎨 Real-time drawing canvas
- 📷 Webcam-based prediction
- 🌐 Flask/Streamlit web application
- 🔡 Support for lowercase letters
- 📝 Word and sentence recognition
- ☁️ Cloud deployment

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- TensorFlow
- Keras
- OpenCV
- TensorFlow Datasets
- EMNIST Dataset

---

## 👩‍💻 Authors

- **Rashmitha Chowdary**
- **S. Nikitha**
- **E. Niharika**
- **K. Nithya**

---

<p align="center">
Made with ❤️ using TensorFlow, OpenCV, and Python
</p>