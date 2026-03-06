from typing import Literal
from pydantic import SecretStr, computed_field, BaseModel
from pydantic_core import MultiHostUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    host: str
    port: int
    user: str
    password: SecretStr
    name: str

    echo: bool | None = None
    pool_size: int = 20
    max_overflow: int = 10
    pool_recycle: int = 1800

    @computed_field
    @property
    def url(self) -> str:
        return MultiHostUrl.build(
            scheme="postgresql+asyncpg",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            path=self.name,
        ).unicode_string()


class LLMSettings(BaseModel):
    """Настройки LLM-провайдера (Gemini)."""

    gemini_api_key: SecretStr
    model_name: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    max_token_size: int = 8192
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    workspace: str = "default"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    env: Literal["dev", "stage", "prod"] = "prod"
    db: DatabaseSettings
    llm: LLMSettings

    @computed_field
    @property
    def db_echo(self) -> bool:
        """SQL echo: явное значение из DB__ECHO или автоматически True для dev."""
        if self.db.echo is not None:
            return self.db.echo
        return self.env == "dev"


settings = AppSettings()