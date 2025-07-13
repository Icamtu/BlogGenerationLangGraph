import logging
import functools
import time
from pathlib import Path
import copy

# ---------------------------
# Logging Configuration Setup
# ---------------------------

# Setup log directory and file path
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_PATH = LOG_DIR / "app.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),                   # Console
        logging.FileHandler(LOG_FILE_PATH, mode='a')  # File
    ]
)

logger = logging.getLogger(__name__)


# -----------------------------------------
# Utility: Redact sensitive values in dicts
# -----------------------------------------

SENSITIVE_KEYS = [
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY"
]

def redact_keys(data: dict, keys_to_redact=None):
    """
    Redact sensitive keys in a dictionary before logging.
    """
    if keys_to_redact is None:
        keys_to_redact = SENSITIVE_KEYS

    redacted = copy.deepcopy(data)
    for key in keys_to_redact:
        if key in redacted:
            redacted[key] = "***REDACTED***"
    return redacted


# -----------------------------------------
# Decorator: Log function entry and exit
# -----------------------------------------

def log_entry_exit(func):
    """
    A decorator that logs function entry, exit, execution time,
    and captures exceptions if any.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"\n{'='*20} ENTER: {func_name} {'='*20}")
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.info(f"{func_name} completed in {execution_time:.4f} seconds")
            logger.info(f"{'='*20} EXIT: {func_name} {'='*21}\n")
            return result

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"Exception in {func_name} after {execution_time:.4f} seconds: {e}", exc_info=True)
            logger.info(f"{'='*20} FAILED: {func_name} {'='*20}\n")
            raise

    return wrapper


# -----------------------------------------
# Example Usage of Redacted Logging
# -----------------------------------------

def log_session_state(session_state: dict):
    redacted_state = redact_keys(session_state)
    logger.info("Session state updated: %s", redacted_state)
