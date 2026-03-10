"""
Tests for macro1/schema/schema.py

Tests the data models including EnvState, Action, AgentState, AgentStatus,
and step/episode data classes.
"""

import pytest
from PIL import Image
from dataclasses import fields

from macro1.schema.schema import (
    EnvState,
    Action,
    AgentState,
    AgentStatus,
    VLMCallingData,
    BaseStepData,
    SingleAgentStepData,
    BaseEpisodeData,
)


class TestEnvState:
    """Tests for EnvState dataclass."""

    def test_create_env_state_with_pixels_only(self, mock_image):
        state = EnvState(pixels=mock_image)
        assert state.pixels == mock_image
        assert state.package is None
        assert state.a11y_tree is None
        assert state.device_time is None

    def test_create_env_state_with_all_fields(self, mock_image):
        state = EnvState(
            pixels=mock_image,
            package="com.example.app",
            a11y_tree={"node": "root"},
            device_time="Thu Dec 4 10:00:00 GMT 2025"
        )
        assert state.pixels == mock_image
        assert state.package == "com.example.app"
        assert state.a11y_tree == {"node": "root"}
        assert state.device_time == "Thu Dec 4 10:00:00 GMT 2025"

    def test_env_state_is_frozen(self, mock_image):
        state = EnvState(pixels=mock_image)
        with pytest.raises(AttributeError):
            state.package = "new.package"


class TestAction:
    """Tests for Action dataclass."""

    def test_create_action_with_name_only(self):
        action = Action(name='press_home')
        assert action.name == 'press_home'
        assert action.parameters is None

    def test_create_action_with_parameters(self):
        action = Action(
            name='click',
            parameters={'point': [540, 960]}
        )
        assert action.name == 'click'
        assert action.parameters == {'point': [540, 960]}

    def test_action_repr(self):
        action = Action(name='click', parameters={'point': [540, 960]})
        repr_str = repr(action)
        assert 'click' in repr_str
        assert 'point' in repr_str

    def test_action_str(self):
        action = Action(name='type', parameters={'text': 'hello'})
        str_val = str(action)
        assert 'type' in str_val
        assert 'text=hello' in str_val

    def test_action_repr_with_none_params(self):
        action = Action(name='press_home')
        assert repr(action) == 'press_home()'

    def test_action_repr_filters_none_values(self):
        action = Action(name='click', parameters={'point': [100, 200], 'extra': None})
        repr_str = repr(action)
        assert 'point' in repr_str
        assert 'None' not in repr_str


class TestAgentState:
    """Tests for AgentState enum."""

    def test_agent_state_values(self):
        assert AgentState.READY.value == 'READY'
        assert AgentState.RUNNING.value == 'RUNNING'
        assert AgentState.CALLUSER.value == 'CALLUSER'

    def test_agent_state_comparison(self):
        state = AgentState.READY
        assert state == AgentState.READY
        assert state != AgentState.RUNNING


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self):
        assert AgentStatus.FINISHED.value == 'FINISHED'
        assert AgentStatus.FAILED.value == 'FAILED'

    def test_agent_status_comparison(self):
        status = AgentStatus.FINISHED
        assert status == AgentStatus.FINISHED
        assert status != AgentStatus.FAILED


class TestVLMCallingData:
    """Tests for VLMCallingData dataclass."""

    def test_create_vlm_calling_data(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        data = VLMCallingData(messages=messages, response="Hello there!")
        assert data.messages == messages
        assert data.response == "Hello there!"


class TestBaseStepData:
    """Tests for BaseStepData dataclass."""

    def test_create_base_step_data(self, mock_env_state):
        step_data = BaseStepData(
            step_idx=0,
            curr_env_state=mock_env_state
        )
        assert step_data.step_idx == 0
        assert step_data.curr_env_state == mock_env_state
        assert step_data.content is None
        assert step_data.action is None
        assert step_data.exec_env_state is None
        assert step_data.vlm_call_history == []

    def test_create_base_step_data_with_action(self, mock_env_state, sample_action):
        step_data = BaseStepData(
            step_idx=1,
            curr_env_state=mock_env_state,
            action=sample_action,
            content="VLM response"
        )
        assert step_data.action == sample_action
        assert step_data.content == "VLM response"


class TestSingleAgentStepData:
    """Tests for SingleAgentStepData dataclass."""

    def test_create_single_agent_step_data(self, mock_env_state):
        step_data = SingleAgentStepData(
            step_idx=0,
            curr_env_state=mock_env_state,
            thought="I need to click the button",
            action_s="click(point=[540, 960])",
            action_desc="Clicking the Photos button",
            answer=None,
            summary="Opened Photos app"
        )
        assert step_data.thought == "I need to click the button"
        assert step_data.action_s == "click(point=[540, 960])"
        assert step_data.summary == "Opened Photos app"

    def test_single_agent_step_data_inherits_from_base(self, mock_env_state):
        step_data = SingleAgentStepData(
            step_idx=0,
            curr_env_state=mock_env_state
        )
        assert hasattr(step_data, 'step_idx')
        assert hasattr(step_data, 'curr_env_state')
        assert hasattr(step_data, 'vlm_call_history')


class TestBaseEpisodeData:
    """Tests for BaseEpisodeData dataclass."""

    def test_create_base_episode_data(self):
        episode = BaseEpisodeData(goal="Open Photos app")
        assert episode.goal == "Open Photos app"
        assert episode.num_steps is None
        assert episode.status is None
        assert episode.message is None
        assert episode.trajectory is None

    def test_create_base_episode_data_with_all_fields(self, mock_env_state, sample_action):
        step = BaseStepData(step_idx=0, curr_env_state=mock_env_state, action=sample_action)
        episode = BaseEpisodeData(
            goal="Complete the task",
            num_steps=5,
            status="FINISHED",
            message="Task completed",
            trajectory=[step]
        )
        assert episode.num_steps == 5
        assert len(episode.trajectory) == 1


class TestDataclassFields:
    """Tests to verify dataclass field definitions."""

    def test_env_state_fields(self):
        field_names = {f.name for f in fields(EnvState)}
        expected = {'pixels', 'package', 'a11y_tree', 'device_time'}
        assert field_names == expected

    def test_action_fields(self):
        field_names = {f.name for f in fields(Action)}
        expected = {'name', 'parameters'}
        assert field_names == expected

    def test_single_agent_step_data_fields(self):
        field_names = {f.name for f in fields(SingleAgentStepData)}
        expected_fields = {
            'step_idx', 'curr_env_state', 'content', 'action',
            'exec_env_state', 'vlm_call_history', 'thought', 'action_s',
            'action_desc', 'answer', 'summary',
        }
        assert expected_fields.issubset(field_names)
