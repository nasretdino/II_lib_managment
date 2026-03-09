from typing import Literal
from pydantic import SecretStr, computed_field, BaseModel, model_validator
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
    """Настройки LLM-провайдера (провайдер-агностичные)."""

    provider: Literal["gemini", "ollama"] = "gemini"
    api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"

    # Модели
    model_name: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    max_token_size: int = 8192

    # Чанкинг (LightRAG)
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    workspace: str = "default"

    # Rate limiting (скользящее окно)
    llm_rate_limit: int = 5          # запросов в окно (для free tier — снизить до 2-3)
    embed_rate_limit: int = 50
    rate_limit_window: float = 60.0

    # Retry при 429 / RESOURCE_EXHAUSTED
    max_retries: int = 5
    llm_retry_base_delay: float = 15.0
    llm_retry_max_delay: float = 90.0
    embed_retry_base_delay: float = 30.0
    embed_retry_max_delay: float = 120.0

    @model_validator(mode="after")
    def validate_provider_settings(self) -> "LLMSettings":
        if self.provider == "gemini" and self.api_key is None:
            raise ValueError("Для провайдера gemini требуется LLM__API_KEY")
        return self


class Neo4jSettings(BaseModel):
    host: str = "neo4j"
    port: int = 7687
    user: str = "neo4j"
    password: SecretStr = SecretStr("neo4jpassword")
    database: str = "neo4j"
    workspace: str = "default"

    @computed_field
    @property
    def uri(self) -> str:
        return f"bolt://{self.host}:{self.port}"


class RagSettings(BaseModel):
    graph_storage: Literal["Neo4JStorage", "NetworkXStorage"] = "Neo4JStorage"


class StorageSettings(BaseModel):
    """Выбор хранилища файлов: local (папка проекта) или minio (S3)."""

    mode: Literal["local", "minio"] = "minio"
    local_path: str = "uploads"


class MinioSettings(BaseModel):
    endpoint: str = "minio:9000"
    access_key: str = "minioadmin"
    secret_key: SecretStr = SecretStr("minioadmin")
    bucket_name: str = "documents"
    use_ssl: bool = False


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
    neo4j: Neo4jSettings = Neo4jSettings()
    rag: RagSettings = RagSettings()
    storage: StorageSettings = StorageSettings()
    minio: MinioSettings = MinioSettings()

    @computed_field
    @property
    def db_echo(self) -> bool:
        """SQL echo: явное значение из DB__ECHO или автоматически True для dev."""
        if self.db.echo is not None:
            return self.db.echo
        return self.env == "dev"


settings = AppSettings()