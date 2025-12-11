# 📊 Customer Churn Prediction
A complete end-to-end machine learning pipeline with preprocessing, model selection, and Streamlit deployment.

## 🌟 Project Overview
This project implements a full Telecom Customer Churn Prediction System using:
- Python
- Scikit-Learn Pipelines
- Feature Engineering (Imputation, Scaling, One-Hot Encoding)
- GridSearchCV Hyperparameter Tuning
- Streamlit Web Application
- Modular, well-structured folder

The workflow trains multiple ML models, selects the best one, and serves predictions in a user-friendly web interface.

## 📁 Project Structure
```
customer_churn/
│
├── data/
│   └── cleaned_data.csv
│
├── models/
│   ├── best_model.joblib
│   ├── feature_columns.json
│   └── <model>_best_model.joblib
│
├── notebooks/
│   ├── cleaning.ipynb
│   └── eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── training.py
│   ├── deployment.py
│   └── app.py
│
└── README.md

```
## ⚙️ Installation & Setup
### 1️⃣ Clone the repository
  git clone https://github.com/yourusername/customer_churn.git
  cd customer_churn
### 2️⃣ Create and activate a virtual environment
  - python -m venv venv
  - venv\Scripts\activate      # Windows
  - source venv/bin/activate   # macOS/Linux
### 3️⃣ Install dependencies
  - pip install -r requirements.txt

## 🧹 Data Preparation
- Data cleaning and EDA are performed in:
    - notebooks/data_cleaning.ipynb
    - notebooks/eda.ipynb
- After cleaning, export the processed dataset to:
    - data/cleaned_data.csv
  The training pipeline expects this file to exist

## 🤖 Model Training
Run training from the project root:
- python -m src.training
- Training will:
    - Load cleaned data
    - Create train/test splits (stratified)
    - Build preprocessing pipeline
    - Train multiple ML models using GridSearchCV
- Save:
    - models/best_model.joblib
    - models/<model>_best_model.joblib
    - models/feature_columns.json
  
  The saved model and feature file ensure consistent prediction during deployment.

## 🖥️ Run Streamlit App
From the project root:
- streamlit run src/app.py
- Features:
    - Single-customer prediction
    - Batch predictions via CSV upload
    - Automatic feature alignment
    - Probability output
    - Clean UI with error handling
  
## 📦 Deployment & Prediction API
- src/deployment.py provides:
  - predict_single(input_dict)   # returns dict with prediction + probability
  - predict_batch(dataframe)     # returns dataframe with predictions appended
- The functions: (Align input to training features)
    - Coerce numeric types
    - Handle missing columns gracefully
    - Ensure stable predictions

## 📊 Technologies Used
- Python 3.10+
- Pandas, NumPy
- Scikit-Learn
- Joblib
- Streamlit
- Jupyter Notebook

## 🚀 Future Enhancements
- SHAP explainability
- FastAPI REST API
- Docker deployment
- Monitoring & model drift detection
- Optuna Bayesian optimization

## 🤝 Contributing
- Contributions, suggestions, and feature requests are welcome.
- Feel free to open an issue or submit a pull request.

## 🙏 Acknowledgements
- Inspired by public telecom churn datasets.
- Thanks to the open-source community for their amazing tools.
