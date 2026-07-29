from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str = ""
    openai_api_key: str = ""  # legacy - voice TTS/STT only
    google_api_key: str = ""  # Gemma 4 via Google AI Studio
    teacher_model: str = "gemma-4-31b-it"
    teacher_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"

    # Local SQLite
    sqlite_path: str = "./echo.db"

    # Gemma 4 - primary local inference lane on port 8003
    gemma4_enabled: bool = True
    gemma4_vllm_base_url: str = "http://127.0.0.1:8003/v1"
    gemma4_base_model: str = "gemma4_e2b"
    gemma4_max_model_len: int = 32768
    gemma4_training_model_path: str = ""

    # Local adapter + training storage
    adapters_dir: str = "./adapters"
    training_data_dir: str = "./training_data"

    # FastAPI
    port: int = 8002
    echo_secret: str = "echo-local-secret"
    demo_seed_token: str = ""

    # JWT Auth
    jwt_secret: str = "echo-jwt-secret-change-in-production"
    token_expire_days: int = 365

    # Routing thresholds
    confidence_threshold: float = 0.70
    min_pairs_for_training: int = 20
    quality_threshold: float = 0.6
    evaluation_holdout_pairs: int = 8
    min_evaluation_pairs: int = 3
    adapter_promotion_min_score: float = 0.25
    adapter_promotion_margin: float = 0.01

    # Teacher policy: Gemma4 is primary; teacher LLM is sparse calibration.
    teacher_policy_enabled: bool = True
    teacher_new_user_pair_threshold: int = 20
    teacher_new_user_daily_cap: int = 10
    teacher_trained_daily_cap: int = 2
    teacher_weekly_cap: int = 5
    legacy_openai_pipeline_enabled: bool = False

    # LiveKit (realtime voice)
    livekit_host: str = "localhost"
    livekit_port: int = 7880
    livekit_api_key: str = "devkey"
    livekit_api_secret: str = "secret"

    # Firebase Cloud Messaging - for background push notifications
    fcm_server_key: str = ""

    # Training runtime portability
    # echo_training_runtime: auto | windows_wsl | linux_local | disabled
    echo_training_runtime: str = "auto"
    # WSL distro used when echo_training_runtime resolves to windows_wsl
    echo_wsl_distro: str = "Ubuntu-24.04"
    # Python executable inside WSL used by Unsloth training
    echo_wsl_training_python: str = "/home/klei/vllm-env/bin/python3"
    # PYTHONPATH inside WSL for Gemma 4-compatible Transformers overrides
    echo_wsl_training_pythonpath: str = "/home/klei/gemma4-transformers"
    # Optional WSL overrides. Empty values use WSL/user defaults and a
    # repository-relative vLLM start script.
    echo_wsl_hf_home: str = ""
    echo_wsl_vllm_start_script: str = ""
    # Override Python used for linux_local training (defaults to sys.executable)
    echo_training_python: str = ""
    # Override PYTHONPATH injected into the linux_local training subprocess
    echo_training_pythonpath: str = ""
    # Override vLLM stop/start commands on linux_local (space-separated, shlex-safe)
    echo_vllm_stop_command: str = ""
    echo_vllm_start_command: str = ""
    # Set true when vLLM is managed externally - skip stop/start around training
    echo_vllm_externally_managed: bool = False

    @property
    def livekit_url(self) -> str:
        return f"ws://{self.livekit_host}:{self.livekit_port}"

    @property
    def llm_api_key(self) -> str:
        # Priority matches TEACHER_BASE_URL: whichever provider is configured wins.
        if "openai.com" in (self.teacher_base_url or ""):
            return self.openai_api_key or self.google_api_key or self.openrouter_api_key
        if "googleapis" in (self.teacher_base_url or ""):
            return self.google_api_key or self.openai_api_key or self.openrouter_api_key
        if "openrouter" in (self.teacher_base_url or ""):
            return self.openrouter_api_key or self.openai_api_key or self.google_api_key
        return self.openai_api_key or self.google_api_key or self.openrouter_api_key


settings = Settings()
