"""
Tests for macro1/schema/config.py

Tests the configuration classes: BaseConfig, VLMConfig,
MobileEnvConfig, AgentConfig, and ReActAgentConfig.
"""

import pytest
import yaml

from macro1.schema.config import (
    BaseConfig,
    MobileEnvConfig,
    VLMConfig,
    AgentConfig,
    ReActAgentConfig,
)


class TestBaseConfig:
    """Tests for BaseConfig class."""

    def test_from_yaml_basic(self, tmp_path):
        class SimpleConfig(BaseConfig):
            name: str = "default"
            value: int = 0

        yaml_content = """
name: test
value: 42
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = SimpleConfig.from_yaml(str(config_path))
        assert config.name == "test"
        assert config.value == 42


class TestMobileEnvConfig:
    """Tests for MobileEnvConfig class."""

    def test_default_values(self):
        config = MobileEnvConfig()
        assert config.serial_no is None
        assert config.host == "127.0.0.1"
        assert config.port == 5037
        assert config.go_home is False
        assert config.wait_after_action_seconds == 2.0

    def test_custom_values(self):
        config = MobileEnvConfig(
            serial_no="emulator-5554",
            host="192.168.1.100",
            port=5038,
            go_home=True,
            wait_after_action_seconds=1.5
        )
        assert config.serial_no == "emulator-5554"
        assert config.host == "192.168.1.100"
        assert config.port == 5038
        assert config.go_home is True
        assert config.wait_after_action_seconds == 1.5

    def test_from_yaml(self, tmp_path):
        yaml_content = """
serial_no: device123
host: localhost
port: 5039
go_home: true
wait_after_action_seconds: 3.0
"""
        config_path = tmp_path / "env_config.yaml"
        config_path.write_text(yaml_content)

        config = MobileEnvConfig.from_yaml(str(config_path))
        assert config.serial_no == "device123"
        assert config.go_home is True


class TestVLMConfig:
    """Tests for VLMConfig class."""

    def test_required_fields(self):
        config = VLMConfig(
            model_name="qwen/qwen3.5-397b-a17b",
            api_key="test-key",
            base_url="https://api.example.com/v1"
        )
        assert config.model_name == "qwen/qwen3.5-397b-a17b"
        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com/v1"

    def test_default_values(self):
        config = VLMConfig(
            model_name="test-model",
            api_key="key",
            base_url="url"
        )
        assert config.max_retry == 3
        assert config.retry_waiting_seconds == 2
        assert config.max_tokens == 1024
        assert config.temperature == 0.0
        assert config.max_pixels == 12845056

    def test_extra_fields_allowed(self):
        config = VLMConfig(
            model_name="test",
            api_key="key",
            base_url="url",
            custom_param="custom_value"
        )
        assert config.custom_param == "custom_value"

    def test_from_yaml(self, tmp_path):
        yaml_content = """
model_name: gpt-4-vision
api_key: secret-key
base_url: https://api.openai.com/v1
max_tokens: 2048
temperature: 0.5
"""
        config_path = tmp_path / "vlm_config.yaml"
        config_path.write_text(yaml_content)

        config = VLMConfig.from_yaml(str(config_path))
        assert config.model_name == "gpt-4-vision"
        assert config.max_tokens == 2048
        assert config.temperature == 0.5


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_default_values(self):
        config = AgentConfig()
        assert config.vlm is None
        assert config.env is None
        assert config.max_steps == 10
        assert config.enable_log is False
        assert config.log_dir is None

    def test_with_sub_configs(self, sample_vlm_config, sample_env_config):
        vlm = VLMConfig(**sample_vlm_config)
        env = MobileEnvConfig(**sample_env_config)

        config = AgentConfig(
            vlm=vlm,
            env=env,
            max_steps=20,
            enable_log=True,
            log_dir="/logs"
        )
        assert config.vlm.model_name == sample_vlm_config['model_name']
        assert config.env.serial_no == sample_env_config['serial_no']
        assert config.max_steps == 20

    def test_from_yaml(self, temp_yaml_config):
        config = AgentConfig.from_yaml(temp_yaml_config)
        assert config.max_steps == 10
        assert config.vlm is not None
        assert config.env is not None


class TestReActAgentConfig:
    """Tests for ReActAgentConfig class."""

    def test_default_values(self):
        config = ReActAgentConfig()
        assert config.num_latest_screenshots == 3
        assert config.max_action_retry == 5
        assert config.prompt_config is None

    def test_custom_values(self):
        config = ReActAgentConfig(
            num_latest_screenshots=5,
            max_action_retry=10,
            max_steps=15
        )
        assert config.num_latest_screenshots == 5
        assert config.max_action_retry == 10
        assert config.max_steps == 15

    def test_inherits_from_agent_config(self):
        config = ReActAgentConfig(max_steps=20, enable_log=True)
        assert config.max_steps == 20
        assert config.enable_log is True

    def test_from_yaml(self, tmp_path):
        yaml_content = """
vlm:
  model_name: test-model
  api_key: key
  base_url: url

max_steps: 15
num_latest_screenshots: 5
max_action_retry: 8
"""
        config_path = tmp_path / "react_config.yaml"
        config_path.write_text(yaml_content)

        config = ReActAgentConfig.from_yaml(str(config_path))
        assert config.max_steps == 15
        assert config.num_latest_screenshots == 5
        assert config.max_action_retry == 8


class TestConfigYAMLRoundTrip:
    """Tests for config YAML serialization and deserialization."""

    def test_full_config_from_yaml(self, tmp_path):
        yaml_content = """
vlm:
  model_name: qwen/qwen3.5-397b-a17b
  api_key: test-api-key
  base_url: https://api.example.com/v1
  max_tokens: 2048
  temperature: 0.1

env:
  serial_no: emulator-5554
  host: 127.0.0.1
  port: 5037
  go_home: false
  wait_after_action_seconds: 2.0

max_steps: 20
enable_log: true
log_dir: /tmp/logs
"""
        config_path = tmp_path / "full_config.yaml"
        config_path.write_text(yaml_content)

        config = AgentConfig.from_yaml(str(config_path))

        assert config.vlm.model_name == "qwen/qwen3.5-397b-a17b"
        assert config.vlm.max_tokens == 2048
        assert config.env.serial_no == "emulator-5554"
        assert config.max_steps == 20
        assert config.enable_log is True
        assert config.log_dir == "/tmp/logs"

    def test_react_agent_config_from_yaml(self, tmp_path):
        yaml_content = """
vlm:
  model_name: qwen/qwen3.5-397b-a17b
  api_key: key
  base_url: url

max_steps: 15
num_latest_screenshots: 4
max_action_retry: 7
"""
        config_path = tmp_path / "react_config.yaml"
        config_path.write_text(yaml_content)

        config = ReActAgentConfig.from_yaml(str(config_path))

        assert config.max_steps == 15
        assert config.num_latest_screenshots == 4
        assert config.max_action_retry == 7
