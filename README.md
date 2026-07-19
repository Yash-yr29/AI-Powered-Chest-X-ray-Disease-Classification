# 🩺 AI Powered Chest X-ray Disease Classification

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-ff4b4b?style=for-the-badge&logo=streamlit)
![CNN](https://img.shields.io/badge/Model-Custom%20CNN-success?style=for-the-badge)
![ResNet18](https://img.shields.io/badge/Transfer%20Learning-ResNet18-orange?style=for-the-badge)

</p>

---

## 📌 Project Overview

This project is one of my first end-to-end Deep Learning applications built to understand how AI can assist in medical image analysis.

The application predicts whether a Chest X-ray belongs to one of the following categories:

- Normal
- Pneumonia
- Tuberculosis

Rather than relying only on pre-trained models, I first designed and trained a **Custom Convolutional Neural Network (CNN)** from scratch to understand how deep learning actually works.

To compare approaches, I also experimented with **ResNet-18 using Transfer Learning** and evaluated its performance against my custom CNN.

Finally, I deployed the complete model using **Streamlit Cloud**, making it accessible through a web application.

---

# 🎯 Motivation

While learning Deep Learning, I realized that simply watching tutorials wasn't enough.

I wanted to build a complete AI project that would teach me:

- Designing CNN architectures
- Training neural networks
- Image preprocessing
- Model evaluation
- Transfer Learning
- Deployment using Streamlit
- Git & GitHub workflow
- Solving real deployment issues

This project became my practical learning journey into Computer Vision.

---

# 🧠 Models Used

## 1️⃣ Custom CNN

Designed and trained from scratch using PyTorch.

Architecture includes:

- Convolution Layers
- ReLU Activation
- Max Pooling
- Fully Connected Layers
- Dropout Regularization

---

## 2️⃣ ResNet-18 (Transfer Learning)

Also experimented with ResNet-18 to compare performance with my custom CNN and understand the advantages of transfer learning.

---

# 📂 Dataset

Dataset Source:

**Kaggle Chest X-ray Dataset**

Classes:

- Normal
- Pneumonia
- Tuberculosis

Images were preprocessed using:

- Resize (224 × 224)
- RGB Conversion
- Tensor Transformation
- Normalization

---

# 🚀 Features

✅ Upload Chest X-ray

✅ AI Disease Prediction

✅ Confidence Score

✅ Professional Streamlit Interface

✅ Cloud Deployment

---

# 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow
- OpenCV
- Matplotlib
- Scikit-learn
- Git
- GitHub

---

# 📊 Model Performance

| Model | Validation Accuracy | Test Accuracy |
|--------|--------------------:|--------------:|
| Custom CNN | 74.91% | 74.62% |
| ResNet-18 | 74.91% | 73.92% |

---

# 📁 Project Structure

```
AI-Powered-Chest-X-ray-Disease-Classification
│
├── app
│   └── app.py
│
├── model
│   └── cnn_chest_xray.pth
│
├── result
│
├── requirements.txt
│
└── README.md
```

---

# 💻 Live Demo

**Streamlit App**

> https://ai-powered-chest-x-ray-disease-classification-dxqnjzfazcdyk9tk.streamlit.app

---


# ⚠ Current Limitation

This model was trained **only on Chest X-ray images**.

Uploading unrelated images (people, animals, vehicles, landscapes, etc.) may produce incorrect predictions because the model has never learned to identify non-medical images.

Future versions will include an additional binary classifier to validate whether the uploaded image is actually a Chest X-ray before disease prediction.

---

# 📚 What I Learned

This project helped me understand:

- Building CNNs from scratch
- Transfer Learning using ResNet-18
- Data preprocessing for Computer Vision
- Image classification pipeline
- Model saving and loading
- Streamlit deployment
- Git & GitHub version control
- Debugging deployment issues
- Cloud deployment using Streamlit Cloud

More importantly, it taught me that building AI isn't just about training models—it's about solving practical engineering problems from development to deployment.

---

# 🚧 Challenges Faced

Some of the major challenges during this project included:

- Understanding CNN architecture
- Loading trained models correctly
- Managing project structure
- Git and GitHub issues
- Streamlit deployment
- Requirements dependency conflicts
- Handling model paths after deployment

Each challenge improved my understanding of real-world AI development.

---

# 🔮 Future Improvements

- Binary Chest X-ray Validation Model
- Improved Dataset
- Better Model Accuracy
-
--

# 👨‍💻 Developer

## Yash Raj

B.Tech Computer Science & Engineering

Indian Institute of Information Technology, Nagpur

GitHub

https://github.com/Yash-yr29

---

# 📜 License

This project is developed for educational and research purposes only.

It is **not intended to replace professional medical diagnosis**.

Always consult a qualified healthcare professional before making medical decisions.
