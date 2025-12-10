"""
Project configuration.

"""

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
DATA_FILE: Path = DATA_DIR / "cleaned_data.csv"

MODEL_DIR: Path = BASE_DIR / "models"
BEST_MODEL_PATH: Path = MODEL_DIR / "best_model.joblib"

# Target configuration (set this to the exact column name in your cleaned CSV)
TARGET_COL: str = "Churn"

# Train/test and CV
TEST_SIZE: float = 0.20
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
SCORING: str = "roc_auc"
N_JOBS: int = -1
