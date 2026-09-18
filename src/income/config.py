from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/baseline_logreg.joblib"
    database_url: str | None = None
    log_level: str = "INFO"

    model_config = {"env_file": ".env"}

settings = Settings()