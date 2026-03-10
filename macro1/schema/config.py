import pydantic
import yaml
from typing import Optional, Union


class BaseConfig(pydantic.BaseModel):
    @classmethod
    def from_yaml(cls, yaml_file: str):
        """Load configuration from a YAML file and create a instance"""
        with open(yaml_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)


class MobileEnvConfig(BaseConfig):
    serial_no: Union[str, None] = None
    host: str = "127.0.0.1"
    port: int = 5037
    go_home: bool = False
    wait_after_action_seconds: float = 2.0


class VLMConfig(BaseConfig):
    model_name: str
    api_key: str
    base_url: str
    max_retry: int = 3
    retry_waiting_seconds: int = 2
    max_tokens: int = 1024
    temperature: float = 0.0
    max_pixels: int = 12845056

    class Config:
        extra = 'allow'


class AgentConfig(BaseConfig):
    vlm: VLMConfig = None
    env: MobileEnvConfig = None
    max_steps: int = 10
    enable_log: bool = False
    log_dir: Optional[str] = None


class ReActAgentConfig(AgentConfig):
    num_latest_screenshots: int = 3
    max_action_retry: int = 5
    prompt_config: str = None
