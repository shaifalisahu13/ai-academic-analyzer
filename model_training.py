import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Drop Student_ID (not useful)
data = data.drop("Student_ID", axis=1)

# Define input and output
X = data.drop("Final Exam Marks (out of 100)", axis=1)
y = data["Final Exam Marks (out of 100)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("R2 Score:", r2_score(y_test, predictions))

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")

