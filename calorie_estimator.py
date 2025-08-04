import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Dummy food-to-calorie mapping (replace with your actual values)
calorie_dict = {
    'pizza': 266,
    'burger': 295,
    'sushi': 200,
    'ice_cream': 207,
    'salad': 150,
    # Add more as needed
}

# Load your trained model (change path if needed)
model = load_model('model/food_model.h5')

# Your class labels (update as per your model training)
class_labels = list(calorie_dict.keys())

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Match model input size
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    food_item = class_labels[predicted_index]
    calories = calorie_dict.get(food_item, "Unknown")
    return food_item, calories

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Food Calorie Estimator")
    parser.add_argument('--image', type=str, required=True, help="Path to food image")
    args = parser.parse_args()

    food, cal = predict(args.image)
    print(f"🍽️ Predicted Food: {food}")
    print(f"🔥 Estimated Calories: {cal} kcal")
