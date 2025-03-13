from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Gemini API settings
GEMINI_MODEL = "gemini-pro-vision"  # for vision tasks
GEMINI_TEXT_MODEL = "gemini-pro"    # for text-only tasks
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")  # API key from environment variable

# Annotation settings
CONFIDENCE_THRESHOLD = os.environ.get("CONFIDENCE_THRESHOLD", 0.7)
MAX_OBJECTS_PER_IMAGE = os.environ.get("MAX_OBJECTS_PER_IMAGE", 20)
MIN_ACTION_COMPLEXITY = os.environ.get("MIN_ACTION_COMPLEXITY", 2)
MIN_OBJECT_COMPLEXITY = os.environ.get("MIN_OBJECT_COMPLEXITY", 2)

# Text generation settings
TEXT_TYPES = {
    "brief": {"max_length": 50},
    "detailed": {"max_length": 150},
    "more_detailed": {"max_length": 300}
}

# Region annotation settings
BOX_CONFIDENCE_THRESHOLD = os.environ.get("BOX_CONFIDENCE_THRESHOLD", 0.5)
NMS_THRESHOLD = os.environ.get("NMS_THRESHOLD", 0.4)

# Blacklist for filtering irrelevant phrases
PHRASE_BLACKLIST = {
    "it", "this", "that", "these", "those",
    "here", "there", "something", "anything",
    "nothing", "everything"
} 