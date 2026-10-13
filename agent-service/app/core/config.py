from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 以当前文件为基准，找向上2级的父目录，也就是项目的根目录
APP_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # 设置.env文件的路径，默认是项目根目录下的.env
    model_config = SettingsConfigDict(
        env_file=(APP_ROOT / ".env",),
        env_file_encoding="utf-8",
        extra="ignore", # 本类中未定义而.env中出现的字段处理方案
    )
    # 应用信息
    app_name: str = "agent-service"
    app_version: str = "0.1.0"
    app_port: int = Field(default=8002, alias="APP_PORT")

    # 日志级别
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="console", alias="LOG_FORMAT")

    # 虚拟机地址
    infra_host: str | None = Field(default=None, alias="INFRA_HOST")

    # 业务端地址
    business_service_url: str | None = Field(default=None, alias="BUSINESS_SERVICE_URL")

    # 数据库配置
    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str | None = Field(default=None, alias="POSTGRES_DB")
    postgres_user: str | None = Field(default=None, alias="POSTGRES_USER")
    postgres_password: str | None = Field(default=None, alias="POSTGRES_PASSWORD")

    # 对话模型
    chat_model: str = Field(default="deepseek-v4-flash", alias="CHAT_MODEL")

    # mineru
    mineru_token: str | None = Field(default=None, alias="MINERU_TOKEN")

    # auth
    jwt_secret: str | None = Field(default=None, alias="JWT_SECRET")

    # DeepSeek配置
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(default="https://api.deepseek.com",alias="DEEPSEEK_BASE_URL")

    # Embedding模型。配置为本地路径时离线加载，不访问 HuggingFace。
    embedding_model_path: str | None = Field(default=None, alias="EMBEDDING_MODEL_PATH")

    # Reranker开关
    rerank_enable: bool = Field(default=False, alias="RERANK_ENABLE")

    # RAG相关参数
    milvus_uri: str | None = Field(default=None, alias="MILVUS_URI")
    milvus_collection: str = Field(
        default="insurance_knowledge_chunks",
        alias="MILVUS_COLLECTION",
    )
    markdown_optimizer_model: str = Field(default="deepseek-v4-flash",alias="MARKDOWN_OPTIMIZER_MODEL")
    rag_child_target_chars: int = Field(default=500, alias="RAG_CHILD_TARGET_CHARS")
    rag_child_overlap_chars: int = Field(default=80, alias="RAG_CHILD_OVERLAP_CHARS")
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3",alias="RERANK_MODEL")
    rag_candidate_k: int = Field(default=5, alias="RAG_CANDIDATE_K")



    @model_validator(mode="after")
    def derive_infra_urls(self) -> "Settings":
        # 拼接数据库地址
        if self.database_url is None and self.infra_host:
            if self.postgres_db and self.postgres_user and self.postgres_password:
                self.database_url = (
                    "postgresql+asyncpg://"
                    f"{self.postgres_user}:{self.postgres_password}"
                    f"@{self.infra_host}:{self.postgres_port}/{self.postgres_db}"
                )
        # 拼接业务端地址
        if self.business_service_url is None and self.infra_host:
            self.business_service_url = f"http://{self.infra_host}:8001"
        # Milvus 使用 HTTP URI；未单独配置时与其它基础设施共用主机。
        if self.milvus_uri is None and self.infra_host:
            self.milvus_uri = f"http://{self.infra_host}:19530"
        return self

settings = Settings()