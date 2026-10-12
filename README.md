# 🏦 Loan Approval Prediction System

### Machine Learning | XGBoost Classification | Flask Web Application

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black?logo=flask)](https://flask.palletsprojects.com/)
[![Deployment](https://img.shields.io/badge/Deployment-Render-purple)](https://render.com/)

**[🚀 Live Demo](https://loan-approval-prediction-xgboost-66ol.onrender.com/)** · **[💻 GitHub Repository](https://github.com/yashchaudhari372-lab/Loan_Approval_Prediction_System)**

## 📌 Project Overview

The **Loan Approval Prediction System** is an end-to-end machine learning project that predicts a loan approval outcome from applicant information. It uses an **XGBoost classification model** and provides predictions through a **Flask-based web interface**.

The project demonstrates a practical ML workflow: preparing applicant data, training and evaluating a classifier, saving the model, and serving predictions in a web application.

> **Important:** This is a demonstration/portfolio project. Its predictions should not be used as the sole basis for real lending decisions. Production lending systems require fairness testing, explainability, privacy protections, regulatory compliance, and human oversight.

## ✨ Key Features

- Loan approval classification using **XGBoost**
- User-facing web interface built with **Flask**
- Saved trained model for inference (`Xg_model.pkl`)
- Hosted live demonstration on **Render**
- End-to-end example of ML model deployment

## 🧰 Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Machine learning | XGBoost classifier |
| Web application | Flask |
| Model persistence | Pickle (`.pkl`) |
| Deployment | Render |
| Additional dependencies | See [`requirements.txt`](requirements.txt) |

## 📊 Model Performance

Previously reported evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | **98%** |
| Weighted F1-score | **0.98** |
| Test samples | **854** |

*These figures are based on the project's previously shared evaluation results; the current repository does not include an independently reproducible evaluation notebook. Confirm the metrics and test split before presenting them as verified benchmarks.*

**Accuracy** measures the proportion of correct predictions. **Weighted F1-score** combines precision and recall while accounting for class support.

## ⚙️ How It Works

```text
Applicant details
       ↓
Input validation and preprocessing
       ↓
Trained XGBoost classifier
       ↓
Predicted approval class
       ↓
Result displayed in Flask web application
```

The exact input fields and preprocessing requirements are defined by the application and trained model. For consistent predictions, inference inputs must use the same feature schema and transformations as model training.

## 📁 Repository Structure

```text
Loan_Approval_Prediction_System/
├── app.py              # Flask application and prediction interface
├── Xg_model.pkl        # Serialized trained model
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yashchaudhari372-lab/Loan_Approval_Prediction_System.git
cd Loan_Approval_Prediction_System
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open the local URL printed in your terminal (often `http://127.0.0.1:5000/`). Enter the requested applicant information and submit the form to view the prediction.

> **Note:** A serialized model may require compatible versions of Python and its ML libraries. Only load `.pkl` files from sources you trust.

## 🌐 Live Application

Try the hosted application:

**🔗 [Loan Approval Prediction — Live Demo](https://loan-approval-prediction-xgboost-66ol.onrender.com/)**

The service may take a moment to wake up if it has been idle.

## 💼 Use Cases

- Educational demonstration of credit-related classification
- Learning model deployment with Flask
- Understanding classification evaluation metrics
- Prototyping decision-support interfaces (not production lending decisions)

## 🔭 Future Improvements

- Publish the training notebook, dataset description, and reproducible evaluation steps
- Document feature definitions, transformations, and train/test methodology
- Add input validation and automated tests
- Add SHAP-based explanations for predictions
- Assess bias, fairness, calibration, and performance across relevant subgroups
- Add monitoring for data drift and model performance
- Improve the web interface and accessibility

## 👨‍💻 Author

**Yash Chaudhari**  
Aspiring Data Scientist & Data Analyst

- **GitHub:** [yashchaudhari372-lab](https://github.com/yashchaudhari372-lab)
- **LinkedIn:** [yashchaudhari08](https://www.linkedin.com/in/yashchaudhari08/)

---

⭐ If you found this project useful, consider starring the repository!
