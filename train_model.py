import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# --- Step 1: Data Preparation with Validation Split ---
# Define the path to the directory containing all food class folders
data_dir = "D:/Downloads/food_project/Dataset/food_images"

# Define image and training parameters
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.2
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 10

# Load the training and validation datasets from a single directory
print("Loading and splitting datasets...")

train_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='training',
    seed=123,
    shuffle=True
)

test_ds = image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    subset='validation',
    seed=123,
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Found {NUM_CLASSES} classes: {class_names}")

# --- Preprocessing and Performance Optimization ---
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2),
])
rescale = keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)


# --- Step 2: Build the Model (Transfer Learning with MobileNetV2) ---
print("\nBuilding the model with a pre-trained MobileNetV2 base...")
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Step 3: Train the Model (Phase 1: Head Training) ---
print(f"\n--- Phase 1: Training the top layers for {EPOCHS_PHASE1} epochs ---")
history_phase1 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1,
    validation_data=test_ds
)

# --- Step 4: Fine-Tuning the Model (Phase 2: Fine-Tuning) ---
base_model.trainable = True
model.compile(
    optimizer=Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"\n--- Phase 2: Fine-tuning the entire model for {EPOCHS_PHASE2} epochs ---")
history_phase2 = model.fit(
    train_ds,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    initial_epoch=EPOCHS_PHASE1,
    validation_data=test_ds
)

# --- Step 5: Save the Final Model ---
model.save("food_recognition_model.h5")
print("\nModel saved as 'food_recognition_model.h5'")

# --- Optional: Visualize Training History ---
def plot_history(history):
    combined_history = {}
    for key in history_phase1.history.keys():
        combined_history[key] = history_phase1.history[key] + history_phase2.history[key]

    acc = combined_history['accuracy']
    val_acc = combined_history['val_accuracy']
    loss = combined_history['loss']
    val_loss = combined_history['val_loss']

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_history(history_phase1)






