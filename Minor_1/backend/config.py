import os

from dotenv import load_dotenv

load_dotenv()

# backend/
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ml folder
ML_DIR = os.path.join(BASE_DIR, "ml")

# data + models
DATA_DIR = os.path.join(ML_DIR, "data")
MODEL_DIR = os.path.join(ML_DIR, "saved_models")

# token
TOKEN_FILE = os.path.join(BASE_DIR, "token.txt")

# env vars
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
