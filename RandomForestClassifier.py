import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Step 1: Simulate medical image data (e.g., pixel values)
# Generate 100 synthetic images, each with 256 features (simulated pixel values)

X = np.random.rand(100, 256)  # 100 samples, 256 features
# Generate random binary labels (0: no tumor, 1: tumor)

y = np.random.randint(0, 2, 100)

# Step 2: Train a machine learning model
# Initialize and train a Random Forest classifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 3: Predict on a new simulated image

new_image = np.random.rand(1, 256)  # Generate a new image with 256 features
prediction = model.predict(new_image)

# Step 4: Print the prediction result

print("Tumor detected!" if prediction[0] == 1 else "No tumor detected.")
