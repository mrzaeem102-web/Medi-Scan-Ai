# Multi-Disease AI Health Assessment Service 🩺

An end-to-end Machine Learning pipeline and FastAPI web service designed to assess health risk profiles across three medical domains simultaneously: **Thyroid Conditions**, **Diabetes**, and **Anemia**.

The system utilizes specialized ensemble tree architectures (XGBoost & LightGBM) and deep neural networks trained on clinical data, integrated into a unified RESTful API for production deployment.

---

## 📌 Features

* **Multi-Disease Risk Analysis**: Dedicated models trained for Thyroid multi-class classification, Diabetes binary classification, and Anemia prediction.
* **Unified API Engine**: Single `/predict/all` endpoint allowing frontends (React, Mobile, etc.) to query multiple health metrics in a single API request.
* **Class Imbalance Handling**: Trained with **SMOTE** oversampling to optimize sensitivity and precision for minority medical classes.
* **Production-Ready FastAPI Backend**: Includes automatic interactive Swagger UI documentation, input verification via Pydantic, and CORS middleware configured for web clients.

---

## 📁 Repository Structure

```text
├── app.py                      # FastAPI application and prediction endpoints
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── Thyroid-Dataset.csv         # Training raw dataset - Thyroid
├── diabetes.csv                # Training raw dataset - Diabetes
├── anemia.csv                  # Training raw dataset - Anemia
│
├── scaler_thyroid.pkl          # Saved StandardScaler for Thyroid features
├── scaler_diabetes.pkl         # Saved StandardScaler for Diabetes features
├── scaler_anemia.pkl           # Saved StandardScaler for Anemia features
├── thyroid_target_encoder.pkl # Saved LabelEncoder for Thyroid target classes
│
├── xgb_thyroid.pkl             # Trained XGBoost Classifier (Thyroid)
├── xgb_diabetes.pkl            # Trained XGBoost Classifier (Diabetes)
└── lgb_anemia.pkl              # Trained LightGBM Classifier (Anemia)