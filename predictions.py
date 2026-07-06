import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("emnist_cnn_fixed.keras")

# Read image
img = cv2.imread("emnist_images/A_32.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Image not found")
    exit()

# --- Preprocess image ---
# Invert if needed (make background black, text white)
if np.mean(img) > 127:
    img = 255 - img

# Threshold to clean binary
_, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Find largest contour and crop
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("No letter detected")
    exit()

cnt = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(cnt)
img = img[y:y+h, x:x+w]

# Add padding and resize to 28x28
padding = 4
img = cv2.copyMakeBorder(img, padding, padding, padding, padding, 
                         cv2.BORDER_CONSTANT, value=0)
img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

# Normalize and reshape
img = img.astype(np.float32) / 255.0
img = img.reshape(1, 28, 28, 1)

# --- Prediction ---
prediction = model.predict(img, verbose=0)
pred = np.argmax(prediction[0])
confidence = prediction[0][pred]
letter = chr(pred + ord('A'))

# Get top 3 predictions
top3_idx = np.argsort(prediction[0])[-3:][::-1]
top3 = [(chr(i + ord('A')), prediction[0][i]) for i in top3_idx]

# --- Display results ---
plt.figure(figsize=(10, 4))

# Processed input image
plt.subplot(1, 2, 1)
plt.imshow(img.reshape(28,28), cmap='gray')
plt.title("Processed Input")
plt.axis('off')

# Top 3 predictions bar chart
plt.subplot(1, 2, 2)
letters = [chr(i + ord('A')) for i in top3_idx]
probs = [prediction[0][i] for i in top3_idx]
colors = ['green' if l == letter else 'gray' for l in letters]
plt.bar(letters, probs, color=colors)
plt.title(f"Top 3 Predictions\nPrediction: {letter} \n Confidence: {confidence:.2%}")
plt.ylabel("Confidence")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()

# Print results
print(f"Prediction: {letter}")
print(f"Confidence: {confidence:.3f}")
print("\nTop 3 predictions:")
for l, prob in top3:
    print(f"  {l}: {prob:.3f}")