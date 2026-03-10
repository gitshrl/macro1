"""
Tests for macro1/environment/mobile_environ.py

Tests the Environment class for interacting with Android devices via ADB + uiautomator2.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from macro1.environment.mobile_environ import Environment, APP_PACKAGES
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
            env, _, mock_u2_device = _make_env(
                mock_adb, mock_u2, go_home=True
            )
            mock_u2_device.press.assert_called_with("home")


class TestAppPackages:
    """Tests for APP_PACKAGES mapping."""

    def test_core_apps_present(self):
        assert "instagram" in APP_PACKAGES
        assert "facebook" in APP_PACKAGES
        assert "tiktok" in APP_PACKAGES
        assert "youtube" in APP_PACKAGES
        assert "twitter" in APP_PACKAGES
        assert "x" in APP_PACKAGES
        assert "whatsapp" in APP_PACKAGES

    def test_twitter_and_x_same_package(self):
        assert APP_PACKAGES["twitter"] == APP_PACKAGES["x"]


class TestEnvironmentRegisterAction:
    """Tests for Environment register_action method."""

    def test_register_custom_action(self):
        p_adb, p_u2 = _env_patches()
        with p_adb as mock_adb, p_u2 as mock_u2:
            env, _, _ = _make_env(mock_adb, mock_u2)

            def custom_action(env, text):
                return f"Custom: {text}"

            env.register_action('custom', custom_action)
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

            with pytest.raises(ValueError, match="Screenshot"):
                env.get_state()


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

    # -- App management --

    def test_execute_open_app_known(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='open_app', parameters={'text': 'instagram'})

        env.execute_action(action)
        mock_u2_device.app_start.assert_called_with('com.instagram.android')

    def test_execute_open_app_unknown_uses_package_name(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='open_app', parameters={'text': 'com.custom.app'})

        env.execute_action(action)
        mock_u2_device.app_start.assert_called_with('com.custom.app')

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

    # -- Coordinate-based interactions (via u2) --

    def test_execute_click_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='click', parameters={'point': (540, 960)})

        env.execute_action(action)
        mock_u2_device.click.assert_called_with(540, 960)

    def test_execute_long_press_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(
            name='long_press',
            parameters={'point': (540, 960), 'time': 3.0}
        )

        env.execute_action(action)
        mock_u2_device.long_click.assert_called_with(540, 960, duration=3.0)

    def test_execute_scroll_direction(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='scroll', parameters={'direction': 'down'})

        env.execute_action(action)
        mock_u2_device.swipe_ext.assert_called_with('down', scale=0.8)

    def test_execute_scroll_coordinates(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='scroll', parameters={
            'start_point': (540, 1500),
            'end_point': (540, 500)
        })

        env.execute_action(action)
        mock_u2_device.swipe.assert_called_with(
            540, 1500, 540, 500, duration=0.5
        )

    # -- Text input (via u2) --

    def test_execute_type_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='type', parameters={'text': 'hello world'})

        env.execute_action(action)
        mock_u2_device.send_keys.assert_called_with('hello world')

    def test_execute_clear_text_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='clear_text', parameters={})

        env.execute_action(action)
        mock_u2_device.clear_text.assert_called_once()

    def test_execute_key_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='key', parameters={'text': 'enter'})

        env.execute_action(action)
        mock_u2_device.press.assert_called_with('enter')

    # -- Navigation (via u2) --

    def test_execute_press_home_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='press_home', parameters={})

        env.execute_action(action)
        mock_u2_device.press.assert_called_with("home")

    def test_execute_press_back_action(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='press_back', parameters={})

        env.execute_action(action)
        mock_u2_device.press.assert_called_with("back")

    def test_execute_wait_action(self, mocked_env):
        env, _, _ = mocked_env
        action = Action(name='wait', parameters={'time': 5.0})

        with patch('macro1.environment.mobile_environ.time.sleep') as mock_sleep:
            env.execute_action(action)
            mock_sleep.assert_any_call(5.0)

    # -- Element-based interactions (via u2) --

    def test_execute_click_by_text(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.click_exists.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_text', parameters={'text': 'OK'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(text='OK', instance=0)

    def test_execute_click_by_id(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.click_exists.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_id',
            parameters={'text': 'com.app:id/button'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(
            resourceId='com.app:id/button', instance=0
        )

    def test_execute_click_by_description(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_el = MagicMock()
        mock_el.click_exists.return_value = True
        mock_u2_device.return_value = mock_el
        action = Action(
            name='click_by_description',
            parameters={'text': 'Navigate up'}
        )

        env.execute_action(action)
        mock_u2_device.assert_called_with(
            description='Navigate up', instance=0
        )

    # -- Screen analysis --

    def test_execute_dump_xml(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_u2_device.dump_hierarchy.return_value = "<hierarchy></hierarchy>"
        action = Action(name='dump_xml', parameters={})

        result = env.execute_action(action)
        mock_u2_device.dump_hierarchy.assert_called_once()
        assert "<hierarchy>" in result

    def test_execute_get_ui_elements(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        mock_u2_device.dump_hierarchy.return_value = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy>
  <node class="android.widget.Button" text="OK" clickable="true"
        bounds="[100,200][300,400]" resource-id="" content-desc="" />
</hierarchy>"""
        action = Action(name='get_ui_elements', parameters={})

        result = env.execute_action(action)
        assert result is not None
        assert "OK" in result

    def test_execute_get_clipboard(self, mocked_env):
        env, mock_device, _ = mocked_env
        mock_device.shell.side_effect = lambda cmd: {
            f"pidof {Environment.__module__.split('.')[0]}": "",  # not used directly
        }.get(cmd, 'Broadcast completed: result=-1, data="copied_text"')
        # Mock _get_clipboard directly since it has internal shell calls
        env._get_clipboard = lambda: "copied_text"
        action = Action(name='get_clipboard', parameters={})

        result = env.execute_action(action)
        assert "copied_text" in result

    # -- Device control --

    def test_execute_open_notification(self, mocked_env):
        env, _, mock_u2_device = mocked_env
        action = Action(name='open_notification', parameters={})

        env.execute_action(action)
        mock_u2_device.open_notification.assert_called_once()

    # -- Terminal actions --

    def test_execute_finished_noop(self, mocked_env):
        env, _, _ = mocked_env
        action = Action(name='finished', parameters={'answer': 'done'})
        env.execute_action(action)  # should not raise

    def test_execute_call_user_noop(self, mocked_env):
        env, _, _ = mocked_env
        action = Action(name='call_user', parameters={'question': 'help?'})
        env.execute_action(action)  # should not raise

    # -- Error handling --

    def test_execute_unknown_action_raises(self, mocked_env):
        env, _, _ = mocked_env
        action = Action(name='unknown_action', parameters={})

        with pytest.raises(ValueError, match="Unknown action"):
            env.execute_action(action)

    def test_execute_registered_custom_action(self, mocked_env):
        env, _, _ = mocked_env

        def custom_func(environment, message):
            return f"Custom: {message}"

        env.register_action('custom', custom_func)
        action = Action(name='custom', parameters={'message': 'test'})

        result = env.execute_action(action)
        assert 'Custom: test' in result


class TestEnvironmentIntegration:
    """Integration tests that require a connected device."""

    @pytest.fixture
    def real_env(self):
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
    def test_01_get_state(self, real_env):
        """Screenshot + current app info."""
        state = real_env.get_state()

        assert isinstance(state, EnvState)
        assert state.pixels is not None
        assert isinstance(state.pixels, Image.Image)
        assert state.package is not None
        assert len(state.package) > 0
        print(f"  Screenshot: {state.pixels.size}, package: {state.package}")

    @pytest.mark.integration
    def test_02_press_home(self, real_env):
        """Go to home screen."""
        action = Action(name='press_home', parameters={})
        real_env.execute_action(action)
        state = real_env.get_state()
        print(f"  On home screen, package: {state.package}")

    @pytest.mark.integration
    def test_03_open_app(self, real_env):
        """Open Settings via open_app."""
        import time
        # Clear all foreground apps (Chrome welcome, hyperclipper overlay, etc.)
        real_env._d.shell("am force-stop com.android.chrome")
        real_env._d.shell("am force-stop id.intiva.hyperclipper")
        real_env.execute_action(Action(name='press_home', parameters={}))
        time.sleep(2)

        action = Action(name='open_app', parameters={'text': 'com.android.settings'})
        real_env.execute_action(action)
        time.sleep(3)
        state = real_env.get_state()
        print(f"  Opened Settings, package: {state.package}")
        assert "settings" in state.package.lower()
        # Go back home
        real_env.execute_action(Action(name='press_home', parameters={}))

    @pytest.mark.integration
    def test_04_click(self, real_env):
        """Tap center of screen."""
        action = Action(name='click', parameters={'point': (540, 960)})
        real_env.execute_action(action)
        print("  Clicked (540, 960)")

    @pytest.mark.integration
    def test_05_scroll_direction(self, real_env):
        """Scroll down then up."""
        real_env.execute_action(Action(name='scroll', parameters={'direction': 'down'}))
        print("  Scrolled down")
        real_env.execute_action(Action(name='scroll', parameters={'direction': 'up'}))
        print("  Scrolled up")

    @pytest.mark.integration
    def test_06_scroll_coordinates(self, real_env):
        """Swipe from bottom to top."""
        action = Action(name='scroll', parameters={
            'start_point': (540, 1500),
            'end_point': (540, 500)
        })
        real_env.execute_action(action)
        print("  Swiped (540,1500) -> (540,500)")

    @pytest.mark.integration
    def test_07_long_press(self, real_env):
        """Long press center."""
        action = Action(name='long_press', parameters={'point': (540, 960), 'time': 1.5})
        real_env.execute_action(action)
        print("  Long pressed (540, 960) for 1.5s")

    @pytest.mark.integration
    def test_08_press_back(self, real_env):
        """Press back button."""
        action = Action(name='press_back', parameters={})
        real_env.execute_action(action)
        print("  Pressed back")

    @pytest.mark.integration
    def test_09_open_notification(self, real_env):
        """Open notification panel."""
        action = Action(name='open_notification', parameters={})
        real_env.execute_action(action)
        print("  Opened notification panel")
        # Close it
        real_env.execute_action(Action(name='press_back', parameters={}))

    @pytest.mark.integration
    def test_10_type_and_clear(self, real_env):
        """Open browser, type text, clear it."""
        import time
        # Go home first, then open browser
        real_env.execute_action(Action(name='press_home', parameters={}))
        real_env.execute_action(Action(name='open_url', parameters={'text': 'https://www.google.com'}))
        time.sleep(3)

        # Click search bar area and type
        real_env.execute_action(Action(name='click', parameters={'point': (540, 300)}))
        time.sleep(2)
        # Try clicking the search box by text if visible
        real_env.execute_action(Action(name='click_by_text', parameters={'text': 'Search or type URL'}))
        time.sleep(1)

        real_env.execute_action(Action(name='type', parameters={'text': 'hello macro1'}))
        print("  Typed 'hello macro1'")
        time.sleep(1)

        # Clear text — may fail on some emulators, that's OK
        try:
            real_env.execute_action(Action(name='clear_text', parameters={}))
            print("  Cleared text")
        except Exception as e:
            print(f"  clear_text skipped: {e}")
            # Fallback: select all + delete via key
            real_env.execute_action(Action(name='key', parameters={'text': 'delete'}))
            print("  Used delete key as fallback")

    @pytest.mark.integration
    def test_11_key_press(self, real_env):
        """Press enter key."""
        action = Action(name='key', parameters={'text': 'enter'})
        real_env.execute_action(action)
        print("  Pressed enter")

    @pytest.mark.integration
    def test_12_open_url(self, real_env):
        """Open a URL in browser."""
        action = Action(name='open_url', parameters={'text': 'https://www.example.com'})
        real_env.execute_action(action)
        state = real_env.get_state()
        print(f"  Opened example.com, package: {state.package}")

    @pytest.mark.integration
    def test_13_get_ui_elements(self, real_env):
        """Get UI elements from current screen."""
        action = Action(name='get_ui_elements', parameters={})
        result = real_env.execute_action(action)
        assert result is not None
        import json
        elements = json.loads(result)
        print(f"  Found {len(elements)} UI elements")
        for el in elements[:5]:
            print(f"    - {el.get('text', '')[:30]} ({el.get('type')}) at {el.get('center')}")

    @pytest.mark.integration
    def test_14_dump_xml(self, real_env):
        """Dump UI hierarchy XML."""
        action = Action(name='dump_xml', parameters={})
        result = real_env.execute_action(action)
        assert result is not None
        assert "<hierarchy" in result or "<node" in result
        print(f"  XML dump: {len(result)} chars")

    @pytest.mark.integration
    def test_15_get_clipboard(self, real_env):
        """Read clipboard via hyperclipper."""
        action = Action(name='get_clipboard', parameters={})
        result = real_env.execute_action(action)
        print(f"  Clipboard: '{result}'")

    @pytest.mark.integration
    def test_16_wait(self, real_env):
        """Wait action."""
        action = Action(name='wait', parameters={'time': 2.0})
        real_env.execute_action(action)
        print("  Waited 2 seconds")

    @pytest.mark.integration
    def test_17_click_by_text(self, real_env):
        """Click element by visible text."""
        real_env.execute_action(Action(name='press_home', parameters={}))
        import time; time.sleep(1)
        # Try clicking something on home screen
        action = Action(name='click_by_text', parameters={'text': 'Google'})
        real_env.execute_action(action)
        print("  Clicked by text 'Google'")

    @pytest.mark.integration
    def test_18_finished_noop(self, real_env):
        """Finished action — no-op on device."""
        action = Action(name='finished', parameters={'answer': 'Integration tests done'})
        real_env.execute_action(action)
        print("  Finished (no-op)")

    @pytest.mark.integration
    def test_19_call_user_noop(self, real_env):
        """Call user action — no-op on device."""
        action = Action(name='call_user', parameters={'question': 'test?'})
        real_env.execute_action(action)
        print("  Call user (no-op)")

    @pytest.mark.integration
    def test_20_press_home_cleanup(self, real_env):
        """Return to home screen after all tests."""
        real_env.execute_action(Action(name='press_home', parameters={}))
        print("  Cleaned up — back to home")
