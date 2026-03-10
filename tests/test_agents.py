"""
Tests for macro1/agents/

Tests the Agent base class and ReActAgent implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from macro1.agents.base import Agent
from macro1.agents.agent_react import ReActAgent, parse_reason_and_action
from macro1.schema.schema import (
    Action, EnvState, AgentState, AgentStatus,
    BaseStepData, SingleAgentStepData,
    BaseEpisodeData,
)


# ============================================
# Helper Functions Tests
# ============================================

class TestParseReasonAndAction:
    """Tests for ReAct agent's parse_reason_and_action function."""

    def test_parse_click_action(self):
        content = """Thought: I need to click on the Photos icon to open it.
Action: click(point=[540, 960])"""

        # Use 1000x1000 so normalization is identity (coords are 0-1000 range)
        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert reason is not None
        assert "click" in reason.lower() or "Photos" in reason
        assert action.name == 'click'
        assert action.parameters['point'] == (540, 960)

    def test_parse_type_action(self):
        content = """Thought: I need to type the search query.
Action: type(text="hello world")"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'type'
        assert action.parameters['text'] == "hello world"

    def test_parse_finished_action(self):
        content = """Thought: The task is complete.
Action: finished(answer="Task done")"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'finished'
        assert action.parameters['answer'] == "Task done"

    def test_parse_invalid_action_raises(self):
        content = "This is not a valid action format"
        raw_size = (1080, 1920)

        with pytest.raises(Exception):
            parse_reason_and_action(content, raw_size)

    def test_parse_open_app_action(self):
        content = """Thought: I need to open Instagram.
Action: open_app(text='instagram')"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'open_app'
        assert action.parameters['text'] == 'instagram'

    def test_parse_scroll_direction_action(self):
        content = """Thought: I need to scroll down to see more content.
Action: scroll(direction='down')"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'scroll'
        assert action.parameters['direction'] == 'down'

    def test_parse_click_by_text_action(self):
        content = """Thought: I see the Login button, I'll click it.
Action: click_by_text(text='Login')"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'click_by_text'
        assert action.parameters['text'] == 'Login'

    def test_parse_action_in_code_block(self):
        content = """Thought: I need to click.
```
click(point=[540, 960])
```"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'click'

    def test_parse_strips_think_tags(self):
        content = """<think>Some internal reasoning here</think>
Thought: I need to press home.
Action: press_home()"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'press_home'

    def test_parse_coordinate_normalization(self):
        """Test that Qwen 0-1000 normalized coordinates are scaled to raw size."""
        content = """Thought: Click center.
Action: click(point=[500, 500])"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'click'
        # 500/1000 * 1080 = 540, 500/1000 * 1920 = 960
        assert action.parameters['point'] == (540, 960)

    def test_parse_get_ui_elements_action(self):
        content = """Thought: The screen is unclear, let me get UI elements.
Action: get_ui_elements()"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'get_ui_elements'

    def test_parse_call_user_action(self):
        content = """Thought: I need help from the user.
Action: call_user(question='What is the password?')"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'call_user'
        assert action.parameters['question'] == 'What is the password?'

    def test_parse_get_clipboard_action(self):
        content = """Thought: I just copied the OTP code, let me read it from clipboard.
Action: get_clipboard()"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'get_clipboard'

    def test_parse_open_url_action(self):
        content = """Thought: I need to open the website.
Action: open_url(text='https://google.com')"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'open_url'
        assert action.parameters['text'] == 'https://google.com'

    def test_parse_long_press_action(self):
        content = """Thought: I need to long press to copy text.
Action: long_press(point=[500, 500])"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'long_press'

    def test_parse_clear_text_action(self):
        content = """Thought: I need to clear the search field.
Action: clear_text()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'clear_text'

    def test_parse_key_action(self):
        content = """Thought: I need to press enter to submit.
Action: key(text='enter')"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'key'
        assert action.parameters['text'] == 'enter'

    def test_parse_press_home_action(self):
        content = """Thought: I need to go back to the home screen.
Action: press_home()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'press_home'

    def test_parse_press_back_action(self):
        content = """Thought: I need to go back.
Action: press_back()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'press_back'

    def test_parse_wait_action(self):
        content = """Thought: The page is loading, I should wait.
Action: wait()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'wait'

    def test_parse_click_by_id_action(self):
        content = """Thought: I'll click the button by its resource ID.
Action: click_by_id(text='com.app:id/submit_btn')"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'click_by_id'
        assert action.parameters['text'] == 'com.app:id/submit_btn'

    def test_parse_click_by_description_action(self):
        content = """Thought: I'll click the search icon by its description.
Action: click_by_description(text='Search')"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'click_by_description'
        assert action.parameters['text'] == 'Search'

    def test_parse_dump_xml_action(self):
        content = """Thought: I need the raw XML to understand the UI.
Action: dump_xml()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'dump_xml'

    def test_parse_open_notification_action(self):
        content = """Thought: I need to check notifications.
Action: open_notification()"""

        raw_size = (1000, 1000)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'open_notification'

    def test_parse_scroll_with_coordinates_action(self):
        content = """Thought: I need to scroll precisely.
Action: scroll(start_point=[500, 800], end_point=[500, 200])"""

        raw_size = (1080, 1920)
        reason, action, action_s = parse_reason_and_action(content, raw_size)

        assert action.name == 'scroll'
        assert 'start_point' in action.parameters
        assert 'end_point' in action.parameters


# ============================================
# Agent Base Class Tests
# ============================================

class TestAgentRegistration:
    """Tests for Agent registration mechanism."""

    def test_react_agent_registered(self):
        assert Agent.by_name('ReAct') is not None

    def test_single_agent_registered(self):
        assert Agent.by_name('SingleAgent') is not None

    def test_react_and_single_agent_are_same(self):
        assert Agent.by_name('ReAct') is Agent.by_name('SingleAgent')


# ============================================
# ReActAgent Tests
# ============================================

class TestReActAgent:
    """Tests for ReActAgent class."""

    @pytest.fixture
    def mock_react_agent(self, mock_environment, mock_vlm_wrapper, mock_env_state):
        """Create a mocked ReActAgent."""
        with patch('macro1.agents.base.Environment', return_value=mock_environment):
            with patch('macro1.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                agent = ReActAgent(max_steps=3)
                agent.env = mock_environment
                agent.vlm = mock_vlm_wrapper

                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = """Thought: I need to click the Photos app.
Action: click(point=[540, 960])"""
                mock_response.choices[0].message.model_extra = {}
                mock_response.model_extra = {}
                mock_vlm_wrapper.predict.return_value = mock_response

                return agent

    def test_react_agent_init(self, mock_react_agent):
        assert mock_react_agent.max_steps == 3
        assert mock_react_agent.state == AgentState.READY

    def test_react_agent_reset(self, mock_react_agent):
        mock_react_agent.reset(goal="Open Photos", max_steps=5)

        assert mock_react_agent.goal == "Open Photos"
        assert mock_react_agent.max_steps == 5
        assert mock_react_agent.curr_step_idx == 0
        assert len(mock_react_agent.trajectory) == 0

    def test_react_agent_reset_initializes_stuck_detection(self, mock_react_agent):
        mock_react_agent.reset(goal="Test")

        assert hasattr(mock_react_agent, '_recent_actions')
        assert mock_react_agent._recent_actions == []
        assert mock_react_agent._max_repeat == 3

    def test_react_agent_step(self, mock_react_agent, mock_env_state):
        mock_react_agent.reset(goal="Open Photos")
        mock_react_agent.env.get_state.return_value = mock_env_state

        step_data = mock_react_agent.step()

        assert step_data is not None
        assert step_data.step_idx == 0
        mock_react_agent.vlm.predict.assert_called()

    def test_react_agent_step_finished(self, mock_react_agent, mock_env_state):
        """Test that finished action sets agent status."""
        mock_react_agent.reset(goal="Test")
        mock_react_agent.env.get_state.return_value = mock_env_state

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """Thought: Task is done.
Action: finished(answer="Done")"""
        mock_response.choices[0].message.model_extra = {}
        mock_response.model_extra = {}
        mock_react_agent.vlm.predict.return_value = mock_response

        step_data = mock_react_agent.step()

        assert mock_react_agent.status == AgentStatus.FINISHED
        assert step_data.answer == "Done"


# ============================================
# Agent State Management Tests
# ============================================

class TestAgentStateManagement:
    """Tests for agent state management."""

    def test_agent_state_transitions(self, mock_environment, mock_vlm_wrapper):
        with patch('macro1.agents.base.Environment', return_value=mock_environment):
            with patch('macro1.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                agent = ReActAgent(max_steps=5)
                agent.env = mock_environment
                agent.vlm = mock_vlm_wrapper

                assert agent.state == AgentState.READY

                agent.reset(goal="Test")
                assert agent.state == AgentState.READY

    def test_set_max_steps(self, mock_environment, mock_vlm_wrapper):
        with patch('macro1.agents.base.Environment', return_value=mock_environment):
            with patch('macro1.agents.base.VLMWrapper', return_value=mock_vlm_wrapper):
                agent = ReActAgent(max_steps=5)
                agent.set_max_steps(10)

                assert agent.max_steps == 10


# ============================================
# Integration Tests
# ============================================

class TestAgentIntegration:
    """Integration tests that require real VLM and device."""

    @pytest.fixture
    def real_react_agent(self):
        import os

        api_key = os.getenv('VLM_API_KEY')
        base_url = os.getenv('VLM_BASE_URL')

        if not api_key or not base_url:
            pytest.skip("VLM credentials not available")

        try:
            import adbutils
            adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
            devices = adb.device_list()
            if not devices:
                pytest.skip("No ADB device connected")
            serial_no = devices[0].serial
        except Exception:
            pytest.skip("Cannot connect to ADB")

        from macro1.schema.config import VLMConfig, MobileEnvConfig

        vlm_config = VLMConfig(
            model_name='qwen/qwen3.5-397b-a17b',
            api_key=api_key,
            base_url=base_url,
            max_tokens=512
        )
        env_config = MobileEnvConfig(
            serial_no=serial_no,
            go_home=False,
            wait_after_action_seconds=1.0
        )

        return ReActAgent(
            vlm=vlm_config,
            env=env_config,
            max_steps=1
        )

    @pytest.mark.integration
    def test_react_agent_single_step(self, real_react_agent):
        real_react_agent.reset(goal="Describe what you see on the screen")

        step_data = real_react_agent.step()

        assert step_data is not None
        assert step_data.curr_env_state is not None
        assert step_data.curr_env_state.pixels is not None
