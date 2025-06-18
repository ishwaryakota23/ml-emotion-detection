import numpy as np
from keras.models import load_model, Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the CNN model
cnn_model = load_model("final_emotion_model.h5")

# Call the model once to initialize input/output tensors
_ = cnn_model.predict(np.random.rand(1, 48, 48, 3))  # assuming RGB input

# Now you can define the feature extractor safely
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)


# Load your dataset
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Normalize image data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Extract deep features using CNN
train_features = feature_extractor.predict(X_train)
test_features = feature_extractor.predict(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(train_features, y_train)

# Evaluate
y_pred = rf_model.predict(test_features)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(rf_model, "rf_emotion_model.pkl")
print("✅ Random Forest model saved as rf_emotion_model.pkl")
