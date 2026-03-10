"""
Tests for macro1/environment/mobile_environ.py

Tests the Environment class for interacting with Android devices via ADB.
Includes both unit tests with mocks and integration tests that require a connected device.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from macro1.environment.mobile_environ import Environment
from macro1.schema.schema import Action, EnvState


def _env_patches():
    """Return stacked patches for adbutils and uiautomator2."""
    return (
        patch('macro1.environment.mobile_environ.adbutils'),
        patch('macro1.environment.mobile_environ.u2'),
    )


def _make_env(mock_adb, mock_u2, **kwargs):
    """Create an Environment with mocked adbutils + u2."""
    mock_device = MagicMock()
    mock_device.window_size.return_value = (1080, 1920)
    mock_device.get_serialno.return_value = "emulator-5554"
    mock_adb.AdbClient.return_value.device.return_value = mock_device

    mock_u2_device = MagicMock()
    mock_u2.connect.return_value = mock_u2_device

    defaults = dict(serial_no='test', go_home=False)
    defaults.update(kwargs)
    env = Environment(**defaults)
    return env, mock_device, mock_u2_device


class TestEnvironmentInit:
    """Tests for Environment initialization."""

    def test_init_with_mocked_device(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, mock_device, _ = _make_env(
                mock_adb, mock_u2, serial_no='emulator-5554'
            )

            assert env.host == '127.0.0.1'
            assert env.port == 5037
            assert env.serial_no == 'emulator-5554'
            assert env.window_size == (1080, 1920)

    def test_init_with_custom_wait_time(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, _, _ = _make_env(
                mock_adb, mock_u2, wait_after_action_seconds=3.5
            )
            assert env.wait_after_action_seconds == 3.5

    def test_init_with_go_home(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, mock_device, _ = _make_env(
                mock_adb, mock_u2, go_home=True
            )
            mock_device.keyevent.assert_called_with("HOME")


class TestEnvironmentActionSpace:
    """Tests for Environment action_space property."""

    def test_action_space_property(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, _, _ = _make_env(mock_adb, mock_u2)

            expected_actions = [
                'open', 'click', 'long_press', 'type', 'key',
                'swipe', 'press_home', 'press_back', 'wait',
                'answer', 'system_button', 'clear_text', 'take_note',
                'open_url', 'push_file', 'install_apk', 'airplane_mode',
                'input_emoticon', 'click_by_text', 'click_by_id',
                'click_by_description', 'dump_xml', 'get_clipboard',
            ]

            for action in expected_actions:
                assert action in env.action_space


class TestEnvironmentRegisterAction:
    """Tests for Environment register_action method."""

    def test_register_custom_action(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, _, _ = _make_env(mock_adb, mock_u2)

            def custom_action(env, text):
                return f"Custom: {text}"

            env.register_action('custom', custom_action)

            assert 'custom' in env.action_space
            assert env._register_function['custom'] == custom_action

    def test_register_non_callable_raises(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, _, _ = _make_env(mock_adb, mock_u2)

            with pytest.raises(ValueError, match="callable"):
                env.register_action('invalid', "not a function")


class TestEnvironmentGetState:
    """Tests for Environment get_state method."""

    def test_get_state_returns_env_state(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_device.screenshot.return_value = Image.new(
                'RGB', (1080, 1920)
            )
            mock_device.app_current.return_value.package = "com.example.app"
            mock_device.shell.return_value = "Thu Dec 4 10:00:00 GMT 2025"
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')
            state = env.get_state()

            assert isinstance(state, EnvState)
            assert state.pixels is not None
            assert state.package == "com.example.app"
            assert state.device_time is not None

    def test_get_state_screenshot_error(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            mock_device = MagicMock()
            mock_device.window_size.return_value = (1080, 1920)
            mock_device.screenshot.side_effect = Exception(
                "Screenshot failed"
            )
            mock_adb.AdbClient.return_value.device.return_value = mock_device

            env = Environment(serial_no='test')

            with pytest.raises(ValueError, match="screenshot"):
                env.get_state()


class TestEnvironmentReset:
    """Tests for Environment reset method."""

    def test_reset_with_go_home(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, mock_device, _ = _make_env(mock_adb, mock_u2)
            mock_device.keyevent.reset_mock()

            env.reset(go_home=True)
            mock_device.keyevent.assert_called_with("HOME")

    def test_reset_without_go_home(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, mock_device, _ = _make_env(mock_adb, mock_u2)
            mock_device.keyevent.reset_mock()

            env.reset(go_home=False)
            mock_device.keyevent.assert_not_called()


class TestEnvironmentExecuteAction:
    """Tests for Environment execute_action method."""

    @pytest.fixture
    def mocked_env(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            with patch('macro1.environment.mobile_environ.time.sleep'):
                env, mock_device, mock_u2_device = _make_env(
                    mock_adb, mock_u2, wait_after_action_seconds=0
                )
                yield env, mock_device, mock_u2_device

    def test_execute_click_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='click', parameters={'coordinate': [540, 960]})

        env.execute_action(action)
        mock_device.click.assert_called_with(540, 960)

    def test_execute_long_press_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='long_press',
            parameters={'coordinate': [540, 960], 'time': 3.0}
        )

        env.execute_action(action)
        mock_device.swipe.assert_called_with(540, 960, 540, 960, duration=3.0)

    def test_execute_type_action_ascii(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='type', parameters={'text': 'hello'})

        with patch('macro1.environment.mobile_environ.time.sleep'):
            env.execute_action(action)
        mock_device.shell.assert_called_with(["input", "text", "hello"])

    def test_execute_swipe_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='swipe', parameters={
            'coordinate': [540, 960],
            'coordinate2': [540, 500]
        })

        env.execute_action(action)
        mock_device.swipe.assert_called_with(
            540, 960, 540, 500, duration=0.5
        )

    def test_execute_press_home_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='press_home', parameters={})

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("HOME")

    def test_execute_press_back_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='press_back', parameters={})

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("BACK")

    def test_execute_key_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(name='key', parameters={'text': 'ENTER'})

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("ENTER")

    def test_execute_wait_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='wait', parameters={'time': 5.0})

        with patch('macro1.environment.mobile_environ.time.sleep') as mock_sleep:
            env.execute_action(action)
            mock_sleep.assert_any_call(5.0)

    def test_execute_open_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='open', parameters={'text': 'com.example.app'})

        env.execute_action(action)
        mock_device.app_start.assert_called_with('com.example.app')

    def test_execute_answer_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='answer', parameters={'text': 'The answer is 42'}
        )

        with patch('macro1.environment.mobile_environ.os.system'):
            result = env.execute_action(action)
            assert result == 'The answer is 42'

    def test_execute_take_note_action(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='take_note', parameters={'text': 'Important note'}
        )

        result = env.execute_action(action)
        assert result == 'Important note'

    def test_execute_system_button_back(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(
            name='system_button', parameters={'button': 'Back'}
        )

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("BACK")

    def test_execute_system_button_home(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(
            name='system_button', parameters={'button': 'Home'}
        )

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("HOME")

    def test_execute_system_button_menu(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(
            name='system_button', parameters={'button': 'Menu'}
        )

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("MENU")

    def test_execute_system_button_enter(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.keyevent.reset_mock()
        action = Action(
            name='system_button', parameters={'button': 'Enter'}
        )

        env.execute_action(action)
        mock_device.keyevent.assert_called_with("ENTER")

    def test_execute_unknown_action_raises(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(name='unknown_action', parameters={})

        with pytest.raises(ValueError, match="Unknown action"):
            env.execute_action(action)

    def test_execute_registered_custom_action(self, mocked_env):
        env, mock_device, _ = mocked_env

        def custom_func(environment, message):
            return f"Custom: {message}"

        env.register_action('custom', custom_func)
        action = Action(name='custom', parameters={'message': 'test'})

        result = env.execute_action(action)
        assert 'Custom: test' in result

    # -- new action tests --

    def test_execute_open_url(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='open_url',
            parameters={'text': 'https://example.com'}
        )

        env.execute_action(action)
        mock_device.shell.assert_called_with([
            "am", "start", "-a", "android.intent.action.VIEW",
            "-d", "https://example.com"
        ])

    def test_execute_airplane_mode_on(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='airplane_mode', parameters={'text': 'on'}
        )

        env.execute_action(action)
        mock_device.shell.assert_called_with([
            "cmd", "connectivity", "airplane-mode", "enable"
        ])

    def test_execute_airplane_mode_off(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='airplane_mode', parameters={'text': 'off'}
        )

        env.execute_action(action)
        mock_device.shell.assert_called_with([
            "cmd", "connectivity", "airplane-mode", "disable"
        ])

    def test_execute_airplane_mode_invalid(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='airplane_mode', parameters={'text': 'maybe'}
        )

        with pytest.raises(ValueError, match="on.*off"):
            env.execute_action(action)

    def test_execute_click_by_text(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.wait.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_text', parameters={'text': 'OK'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(text='OK', instance=0)
        mock_el.click.assert_called_once()

    def test_execute_click_by_id(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.wait.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_id',
            parameters={'text': 'com.app:id/button'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(
            resourceId='com.app:id/button', instance=0
        )
        mock_el.click.assert_called_once()

    def test_execute_click_by_description(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.wait.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_description',
            parameters={'text': 'Navigate up'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(
            description='Navigate up', instance=0
        )
        mock_el.click.assert_called_once()

    def test_execute_dump_xml(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_u2_device.dump_hierarchy.return_value = "<hierarchy></hierarchy>"
        action = Action(name='dump_xml', parameters={})

        result = env.execute_action(action)
        mock_u2_device.dump_hierarchy.assert_called_once()
        assert "<hierarchy>" in result

    def test_execute_install_apk(self, mocked_env):
        env, mock_device, _ = mocked_env
        action = Action(
            name='install_apk', parameters={'text': '/tmp/app.apk'}
        )

        env.execute_action(action)
        mock_device.install.assert_called_with('/tmp/app.apk')

    def test_execute_get_clipboard(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.shell.side_effect = [
            "",  # am start hyperclipper
            "1234",  # pidof
            'data="hello world"',  # am broadcast -a clipper.get
            "",  # clipper.set clear
            "",  # force-stop
        ]
        action = Action(name='get_clipboard', parameters={})

        result = env.execute_action(action)
        assert result == "hello world"


class TestEnvironmentIntegration:
    """Integration tests for Environment that require a connected device.

    These tests are skipped if no ADB device is connected.
    """

    @pytest.fixture
    def real_env(self):
        """Create an Environment with actual device connection."""
        try:
            import adbutils
            adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
            devices = adb.device_list()
            if not devices:
                pytest.skip("No ADB device connected")

            serial_no = devices[0].serial
            return Environment(
                serial_no=serial_no,
                host="127.0.0.1",
                port=5037,
                go_home=False,
                wait_after_action_seconds=1.0
            )
        except Exception as e:
            pytest.skip(f"Cannot connect to ADB: {e}")

    @pytest.mark.integration
    def test_get_state_real_device(self, real_env):
        state = real_env.get_state()

        assert isinstance(state, EnvState)
        assert state.pixels is not None
        assert isinstance(state.pixels, Image.Image)
        assert state.package is not None
        assert len(state.package) > 0

    @pytest.mark.integration
    def test_execute_click_real_device(self, real_env):
        action = Action(name='click', parameters={'coordinate': [540, 960]})
        real_env.execute_action(action)

    @pytest.mark.integration
    def test_reset_real_device(self, real_env):
        real_env.reset(go_home=True)

        state = real_env.get_state()
        assert state.package is not None

    @pytest.mark.integration
    def test_action_space_contains_all(self, real_env):
        expected = ['click', 'type', 'swipe', 'press_home', 'press_back']
        for action in expected:
            assert action in real_env.action_space
