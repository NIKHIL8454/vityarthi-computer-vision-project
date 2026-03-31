import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

print("Program started")

img_path = r"C:\Users\SHAUN\OneDrive\Desktop\Computer Vision Vityarthi Project\ants\ants (1).jpg"

if not os.path.exists(img_path):
    print("Error: Image not found")
    exit()

print("Loading image...")

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

print("Image ready")

print("Loading model...")

model = load_model("pest_classifier_model.h5")

print("Model loaded")

prediction = model.predict(img_array)

class_names = [
    'ants','bees','beetle','caterpillar',
    'earthworms','earwig','grasshopper',
    'moth','slug','snail','wasp','weevil'
]

confidence = np.max(prediction)
predicted_class = class_names[np.argmax(prediction)]

print("\n--- RESULT ---")

threshold = 0.6

if confidence < threshold:
    print("Result: Not a pest")

else:
    print("Prediction:", predicted_class)
    print("Confidence:", float(confidence))

    print("\n--- DAMAGE / SYMPTOMS ---")

    info_path = f"pest_info/{predicted_class}.txt"

    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            print(f.read())
    else:
        print("No information available.")

    print("\n--- SOLUTIONS / REMEDIES ---")

    solution_path = f"pest_solution/{predicted_class}.txt"

    if os.path.exists(solution_path):
        with open(solution_path, "r") as f:
            print(f.read())
    else:
        print("No solution available.")

print("\n--- TOP 3 PREDICTIONS ---")

top3_idx = prediction[0].argsort()[-3:][::-1]

for i in top3_idx:
    print(f"{class_names[i]} : {prediction[0][i]:.4f}")
