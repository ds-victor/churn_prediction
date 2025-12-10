"""
src package for Customer Churn project.

Modules:
- config: configuration and paths
- preprocessing: data loading, splitting and transformers (imputer, scaler, encoder)
- training: model training with GridSearchCV
- deployment: model load and predict helpers
- app: Streamlit application

Run training:
    python -m src.training

Run app:
    streamlit run src.app
"""
__all__ = ["config", "preprocessing", "training", "deployment", "app"]
