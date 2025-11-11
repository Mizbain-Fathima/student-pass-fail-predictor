# 🎓 Student Pass/Fail Predictor

An interactive Machine Learning web app that predicts whether a student is likely to Pass or Fail based on their academic performance, personal, and socio-economic factors.

---

Built using:

- Scikit-learn — for training the Decision Tree Classifier
- Pandas — for data handling and preprocessing
- Joblib — for saving and loading the trained model
- Streamlit — for creating a clean, interactive web UI
- Live Demo

Once deployed on Streamlit Cloud, you can access it here (replace with your link):

👉 https://student-pass-fail-predictor.streamlit.app

## About the Project

This app is based on the Student Performance Dataset.
The dataset contains detailed information about Portuguese secondary school students, including:

- Demographics (age, gender, address)
- Academic background (previous grades G1, G2)
- Family and social factors (parents’ jobs, education, relationships)
- Lifestyle and study habits (study time, failures, absences)

The model predicts a binary outcome:

PASS → final grade G3 >= 10
FAIL → final grade G3 < 10

## Features

- Predicts Pass/Fail status instantly
- Displays confidence score (%) for prediction
- User-friendly, interactive web interface
- Built entirely with open-source tools
- Can be accessed anywhere using Streamlit Cloud

## Tech Stack

| Component         | Technology               |
| ----------------- | ------------------------ |
| Model             | Decision Tree Classifier |
| Framework         | Scikit-learn             |
| Frontend          | Streamlit                |
| Data Handling     | Pandas                   |
| Model Persistence | Joblib                   |

## Folder Structure

```bash
📦 Student-performance-prediction
 ┣  app_pass_fail.py              # Streamlit web app
 ┣  student_pass_fail_model.py    # Training script for classifier
 ┣  best_pass_fail_model.joblib   # Saved trained ML model
 ┣  student-performance.csv       # Dataset (optional)
 ┣  requirements.txt              # Dependencies for deployment
 ┗  README.md                     # Project documentation
```

##  How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Mizbain-Fathima/student-pass-fail-predictor.git
cd student-pass-fail-predictor
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # For Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app_pass_fail.py
```

Your app will start locally at:
http://localhost:8501

## Deploying on Streamlit Cloud

Push your project to a public GitHub repo.

Go to https://share.streamlit.io

Click “New App” → Select your repo and branch.

Set Main file path: app_pass_fail.py

Click Deploy 

Your model and app will be hosted at a public URL like:
https://student-pass-fail-predictor.streamlit.app

## Model Summary

| Metric              | Value                        |
| ------------------- | ---------------------------- |
| **Model Type**      | Decision Tree Classifier     |
| **Best Depth**      | Tuned via GridSearchCV       |
| **Accuracy (Test)** | ~85–90% (depending on split) |
| **Top Features**    | G2, G1, absences, failures   |
