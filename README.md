# 🍽️ Food Recognition & Calorie Estimation using CNN

This project uses the Food-101 dataset to train a deep learning model that predicts the **food item** and estimates its **calories per serving** using Convolutional Neural Networks (CNN).

## 🚀 Project Goals
- Recognize food items from images 📸
- Estimate calorie count 🔥
- Enable real-time predictions 🧠

---

## 🧰 Tools & Technologies
- Python
- TensorFlow / Keras
- OpenCV
- Matplotlib, NumPy
- Food-101 Dataset

---

## 📂 Dataset
- [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
- 101 food categories | 101,000 images

---

## 🔍 Features
- CNN model trained to classify 101 food items
- Custom calorie estimation mapping
- Predict any food image in real time
- Easy-to-use script: `predict.py`

---

## 📸 Sample Prediction

<img src="prediction/sample_output.png" width="400" />

```bash
$ python predict.py --image fried_rice.jpg
Predicted: Fried Rice
Estimated Calories: 163 kcal
