import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset (for average comparison)
data = pd.read_csv("dataset.csv")
dataset_average = data["Final Exam Marks (out of 100)"].mean()

# Page title
st.title("🎓 Student Performance Insight Dashboard")

st.markdown(
    "This intelligent system predicts final exam marks and provides performance insights."
)

st.divider()

# Sidebar Inputs
st.sidebar.header("📌 Enter Student Details")

attendance = st.sidebar.number_input("Attendance (%)", 0.0, 100.0)
internal1 = st.sidebar.number_input("Internal Test 1 (out of 40)", 0.0, 40.0)
internal2 = st.sidebar.number_input("Internal Test 2 (out of 40)", 0.0, 40.0)
assignment = st.sidebar.number_input("Assignment Score (out of 10)", 0.0, 10.0)
study_hours = st.sidebar.number_input("Daily Study Hours", 0.0, 24.0)

# Predict Button
if st.sidebar.button("Predict Performance"):

    # ===============================
    # 🎯 Prediction
    # ===============================

    features = np.array([[attendance, internal1, internal2, assignment, study_hours]])
    prediction = model.predict(features)
    predicted_marks = round(prediction[0], 2)

    st.subheader("📊 Predicted Final Exam Score")

    st.progress(int(predicted_marks))
    st.metric("Expected Final Marks", f"{predicted_marks} / 100")

    # Performance Category
    if predicted_marks >= 75:
        st.success("🌟 Excellent Performance! You are academically strong and consistent.")
    elif predicted_marks >= 50:
        st.info("👍 Moderate Performance. With better consistency, you can improve significantly.")
    else:
        st.warning("📚 Improvement Needed. Focus on internal tests and regular study hours.")

    # ===============================
    # 📈 Student vs Dataset Average
    # ===============================

    st.divider()
    st.subheader("📈 Performance Compared to Overall Average")

    difference = round(predicted_marks - dataset_average, 2)

    if difference > 0:
        st.success(f"You are performing {difference} marks ABOVE the overall student average.")
    else:
        st.error(f"You are {abs(difference)} marks BELOW the overall student average.")

    # ===============================
    # 📊 Academic Strength Profile (Horizontal)
    # ===============================

    st.divider()
    st.subheader("📊 Academic Strength Profile")

    # Normalize everything to percentage (0–100 scale)
    normalized_data = {
        "Attendance": attendance,
        "Internal Test 1": (internal1 / 40) * 100,
        "Internal Test 2": (internal2 / 40) * 100,
        "Assignment": (assignment / 10) * 100,
        "Study Hours": (study_hours / 24) * 100
    }

    profile_df = pd.DataFrame({
        "Component": list(normalized_data.keys()),
        "Strength (%)": list(normalized_data.values())
    }).sort_values(by="Strength (%)")

    st.caption("Normalized academic performance across all input components.")

    st.bar_chart(profile_df.set_index("Component"))

    # ===============================
    # 🧠 Personalized Improvement Suggestions
    # ===============================

    st.divider()
    st.subheader("🧠 Personalized Improvement Suggestions")

    if attendance < 75:
        st.write("• Improve attendance to enhance conceptual clarity and internal marks.")
    if internal1 < 25 or internal2 < 25:
        st.write("• Strengthen preparation strategy for internal assessments.")
    if study_hours < 3:
        st.write("• Increase daily study time for better retention and performance.")
    if assignment < 7:
        st.write("• Focus on assignment quality to secure easy scoring marks.")

    if attendance >= 75 and internal1 >= 25 and internal2 >= 25 and study_hours >= 3:
        st.success("You are maintaining a balanced academic profile. Keep it up!")
