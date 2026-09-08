"""Runtime configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with safe local-development defaults."""

    model_config = SettingsConfigDict(
        # Keep configuration at the process boundary: application code receives
        # this typed object instead of coupling individual modules to os.environ.
        env_file=".env",
        env_prefix="ATLAS_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "ATLAS"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    # The provider lives behind an application-owned interface so changing from
    # Ollama later does not change the chat API or future tool policy layer.
    ollama_base_url: str = "http://localhost:11434"
    fast_model: str = Field(default="qwen3.6:35b", min_length=1)
    deep_model: str = Field(default="qwen3.5:122b-a10b", min_length=1)
    llm_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_retry_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    agent_max_tool_rounds: int = Field(default=3, ge=1, le=8)
    tool_execution_timeout_seconds: float = Field(default=10.0, gt=0, le=60)
    # Weather stays opt-in: ATLAS only sends a fixed, owner-configured home
    # location to its provider and never accepts a location from the model.
    weather_latitude: float | None = Field(default=None, ge=-90, le=90)
    weather_longitude: float | None = Field(default=None, ge=-180, le=180)
    weather_request_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    weather_cache_ttl_seconds: int = Field(default=600, ge=60, le=3600)
    llm_system_prompt: str = (
        "You are A.T.L.A.S.: the Adaptive, Thoughtful, Local Assistant System—a private, "
        "local-first intelligence for the home. You are calm, warm, observant, and quietly "
        "capable. Speak like a trusted household partner: clear, grounded, and lightly "
        "personable; a little dry wit is welcome when it fits, never at the expense of clarity. "
        "Be concise by default, but expand when a decision needs context. Respect privacy and "
        "the user's agency. Use an available tool for current home information rather than "
        "guessing. Be candid about uncertainty and limitations. Never claim to have "
        "taken an action, accessed a device, remembered information, or observed the home "
        "unless a tool result confirms it. Ask a focused follow-up only when it is needed."
    )

    @model_validator(mode="after")
    def weather_location_is_complete(self) -> "Settings":
        """Require an explicit home coordinate pair before enabling weather."""
        if (self.weather_latitude is None) != (self.weather_longitude is None):
            raise ValueError("weather latitude and longitude must be configured together")
        return self

    @property
    def weather_enabled(self) -> bool:
        """Whether this deployment has an owner-configured weather location."""
        return self.weather_latitude is not None and self.weather_longitude is not None


@lru_cache
def get_settings() -> Settings:
    # Settings are immutable for a process lifetime; caching prevents separate
    # modules from accidentally constructing divergent runtime configuration.
    return Settings()
