# Customer Churn Prediction

A Streamlit web app that predicts whether a bank customer is likely to churn, using an Artificial Neural Network (ANN) built with TensorFlow/Keras.

## Overview

Given basic customer details (credit score, geography, gender, age, tenure, balance, etc.), the app returns a churn probability and a plain-language verdict ("likely to leave" / "not likely to leave").

## Dataset

Trained on `Churn_Modelling.csv` — a bank customer dataset with the following features:

- CreditScore, Geography, Gender, Age, Tenure, Balance
- NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- Target: `Exited` (1 = churned, 0 = retained)

## Project Structure

| File | Description |
|---|---|
| `experiments.ipynb` | Data preprocessing, EDA, and model training |
| `prediction.ipynb` | Loading the saved model/encoders and running sample predictions |
| `churn_model.h5` | Trained Keras ANN model |
| `scaler.pkl` | StandardScaler fitted on training features |
| `label_gender_encoder.pkl` | LabelEncoder for the `Gender` column |
| `onehot_encoder_geo.pkl` | OneHotEncoder for the `Geography` column |
| `app.py` | Streamlit app for interactive predictions |
| `requirments.txt` | Python dependencies |

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/18091998/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirments.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open in your browser. Enter customer details in the sidebar/form and get a real-time churn prediction.

## Tech Stack

- **Model:** TensorFlow / Keras (Artificial Neural Network)
- **Preprocessing:** scikit-learn (StandardScaler, OneHotEncoder, LabelEncoder)
- **App/UI:** Streamlit
- **Data handling:** pandas, NumPy

## License

This project is licensed under the GPL-3.0 License — see the [LICENSE](LICENSE) file for details.
