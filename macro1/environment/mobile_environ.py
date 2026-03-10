import os
import re
import time
import base64
import logging
import traceback
from typing import Optional

import adbutils
import uiautomator2 as u2
from macro1.schema.schema import Action, EnvState
from macro1.utils.utils import contains_non_ascii

logger = logging.getLogger(__name__)

APP_PACKAGES = {
    "instagram": "com.instagram.android",
    "facebook": "com.facebook.katana",
    "tiktok": "com.zhiliaoapp.musically",
    "youtube": "com.google.android.youtube",
    "twitter": "com.twitter.android",
    "x": "com.twitter.android",
    "whatsapp": "com.whatsapp",
    "telegram": "org.telegram.messenger",
    "chrome": "com.android.chrome",
    "snapchat": "com.snapchat.android",
    "discord": "com.discord",
    "spotify": "com.spotify.music",
    "settings": "com.android.settings",
    "camera": "com.android.camera2",
    "gmail": "com.google.android.gm",
    "google maps": "com.google.android.apps.maps",
    "maps": "com.google.android.apps.maps",
    "play store": "com.android.vending",
    "messenger": "com.facebook.orca",
    "line": "jp.naver.line.android",
    "pinterest": "com.pinterest",
    "reddit": "com.reddit.frontpage",
    "linkedin": "com.linkedin.android",
    "shopee": "com.shopee.id",
    "tokopedia": "com.tokopedia.tkpd",
    "gojek": "com.gojek.app",
    "grab": "com.grabtaxi.passenger",
    "dana": "id.dana",
    "ovo": "ovo.id",
    "gopay": "com.gojek.app",
}

HYPERCLIPPER_PKG = "id.intiva.hyperclipper"
HYPERCLIPPER_ACTIVITY = f"{HYPERCLIPPER_PKG}/.MainActivity"
SELECTOR_TIMEOUT = 5


class Environment:
    def __init__(
        self,
        serial_no: str = None,
        host: str = "127.0.0.1",
        port: int = 5037,
        go_home: bool = False,
        wait_after_action_seconds: float = 2.0
    ):
        self.host = host
        self.port = port
        self.serial_no = serial_no
        self.wait_after_action_seconds = wait_after_action_seconds

        self._action_space = [
            'open', 'open_app', 'click', 'long_press', 'type', 'key',
            'swipe', 'press_home', 'press_back', 'wait',
            'answer', 'system_button', 'clear_text', 'take_note',
            'open_url', 'push_file', 'install_apk', 'airplane_mode',
            'input_emoticon', 'click_by_text', 'click_by_id',
            'click_by_description', 'dump_xml', 'get_clipboard',
            'fetch_otp',
        ]
        self._register_function = {}

        self._d = self._setup_device(serial_no, host, port)
        self._u2 = self._setup_u2(serial_no)
        self.reset(go_home=go_home)
        self.window_size = self._d.window_size(landscape=False)

    def _setup_device(self, serial_no: str, host: str, port: int):
        try:
            adb = adbutils.AdbClient(host=host, port=port)
            device = adb.device(serial_no)
        except Exception as e:
            logger.error(f"Failed to connect to the device: {serial_no}.")
            raise e
        return device

    def _setup_u2(self, serial_no: str):
        try:
            device = u2.connect(serial_no)
        except Exception as e:
            logger.error(f"Failed to connect u2 to device: {serial_no}.")
            raise e
        return device

    def reset(self, go_home: bool = False):
        if go_home:
            self._d.keyevent("HOME")

    def get_state(self, display_id: int = -1) -> EnvState:
        try:
            pixels = self._d.screenshot(display_id, error_ok=False)
        except Exception as e:
            raise ValueError(
                f"Get screenshot error, {traceback.format_exc()}"
            ) from e

        package = self._d.app_current().package
        device_time = self._d.shell('date')
        state = EnvState(
            pixels=pixels, package=package, device_time=device_time
        )
        return state

    @property
    def action_space(self):
        return self._action_space

    def register_action(self, action_name: str, action_func):
        if action_name in self.action_space:
            logger.warning(
                f"Action {action_name} is already registered. Overwriting it."
            )
        if not callable(action_func):
            raise ValueError(
                f"Action function for {action_name} must be callable."
            )

        self._action_space.append(action_name)
        self._register_function[action_name] = action_func

    # -- hyperclipper helpers --

    def _ensure_hyperclipper(self):
        self._d.shell(f"am start -n {HYPERCLIPPER_ACTIVITY}")
        for _ in range(10):
            pid = self._d.shell(
                f"pidof -s {HYPERCLIPPER_PKG}"
            ).strip()
            if pid:
                return True
            time.sleep(1)
        logger.warning("Hyperclipper did not start in time.")
        return False

    def _stop_hyperclipper(self):
        self._d.shell(f"am force-stop {HYPERCLIPPER_PKG}")

    # -- new actions --

    def _open_url(self, url: str):
        self._d.shell([
            "am", "start", "-a", "android.intent.action.VIEW", "-d", url
        ])

    def _push_file(
        self, local_path: str,
        destination: str = "/storage/emulated/0/DCIM"
    ):
        self._d.sync.push(local_path, destination)
        self._d.shell([
            "content", "call", "--method", "scan_volume",
            "--uri", "content://media", "--arg", "external_primary"
        ])
        logger.info(f"Pushed {local_path} to {destination}, media refreshed.")

    def _install_apk(self, apk_path: str):
        self._d.install(apk_path)
        logger.info(f"Installed {apk_path}.")

    def _airplane_mode(self, state: str):
        if state == "on":
            self._d.shell(["cmd", "connectivity", "airplane-mode", "enable"])
        elif state == "off":
            self._d.shell(["cmd", "connectivity", "airplane-mode", "disable"])
        else:
            raise ValueError(
                f"airplane_mode expects 'on' or 'off', got '{state}'"
            )

    def _input_emoticon(self, text: str):
        self._ensure_hyperclipper()
        self._d.shell([
            "am", "broadcast", "-a", "clipper.set", "-e", "text", text
        ])
        self._d.shell(["input", "keyevent", "279"])
        time.sleep(0.5)
        self._d.keyevent("BACK")
        self._d.shell(["input", "keyevent", "279"])
        self._stop_hyperclipper()

    def _click_by_text(self, text: str, index: int = 0):
        el = self._u2(text=text, instance=index)
        if el.wait(timeout=SELECTOR_TIMEOUT):
            el.click()
        else:
            logger.warning(f"Text element not found: {text}")

    def _click_by_id(self, resource_id: str, index: int = 0):
        el = self._u2(resourceId=resource_id, instance=index)
        if el.wait(timeout=SELECTOR_TIMEOUT):
            el.click()
        else:
            logger.warning(f"Resource ID not found: {resource_id}")

    def _click_by_description(self, desc: str, index: int = 0):
        el = self._u2(description=desc, instance=index)
        if el.wait(timeout=SELECTOR_TIMEOUT):
            el.click()
            return
        el2 = self._u2(descriptionContains=desc, instance=index)
        if el2.wait(timeout=SELECTOR_TIMEOUT):
            el2.click()
        else:
            logger.warning(f"Description not found: {desc}")

    def _dump_xml(self) -> str:
        return self._u2.dump_hierarchy()

    def _fetch_otp(self, port: str) -> Optional[str]:
        import urllib.request
        import json
        try:
            url = f"http://127.0.0.1:{port}/sms/latest-otp"
            req = urllib.request.Request(url, method='GET')
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                otp = data.get('otp') or data.get('code') or data.get('message')
                logger.info(f"Fetched OTP from port {port}: {otp}")
                return str(otp) if otp else None
        except Exception as e:
            logger.warning(f"Failed to fetch OTP from port {port}: {e}")
            return None

    def _get_clipboard(self) -> Optional[str]:
        if not self._ensure_hyperclipper():
            return None
        result = self._d.shell("am broadcast -a clipper.get")
        self._d.shell([
            "am", "broadcast", "-a", "clipper.set", "-e", "text", "-"
        ])
        self._stop_hyperclipper()
        match = re.search(r'data="([^"]+)"', result)
        if match:
            return match.group(1)
        logger.warning(f"Could not parse clipboard: {result}")
        return None

    # -- action dispatch --

    def execute_action(self, action: Action) -> Optional[str]:
        result = None

        if action.name in self._register_function:
            result = self._register_function[action.name](
                self, **action.parameters
            )
        else:
            match action.name:
                case 'open':
                    package_name = action.parameters['text']
                    self._d.app_start(package_name)

                case 'open_app':
                    app_name = action.parameters['text'].strip().lower()
                    package = APP_PACKAGES.get(app_name)
                    if package:
                        logger.info(f"Opening app: {app_name} -> {package}")
                        self._d.app_start(package)
                    else:
                        logger.warning(f"Unknown app '{app_name}', trying as package name")
                        self._d.app_start(app_name)

                case 'click':
                    x, y = action.parameters.get('point') or action.parameters.get('coordinate')
                    self._d.click(x, y)

                case 'long_press':
                    x, y = action.parameters.get('point') or action.parameters.get('coordinate')
                    duration = action.parameters.get('time', 2.0)
                    self._d.swipe(x, y, x, y, duration=duration)

                case 'type':
                    text = action.parameters['text']

                    if contains_non_ascii(text):
                        logger.info(
                            "Using ADB keyboard to input non-ASCII text."
                        )
                        charsb64 = str(
                            base64.b64encode(text.encode('utf-8'))
                        )[1:]
                        self._d.shell([
                            "ime", "enable",
                            'com.android.adbkeyboard/.AdbIME'
                        ])
                        self._d.shell([
                            "ime", "set",
                            'com.android.adbkeyboard/.AdbIME'
                        ])
                        time.sleep(1)
                        os.system(
                            f"adb -P {self.port} "
                            f"-s {self._d.get_serialno()} "
                            f"shell am broadcast -a ADB_INPUT_B64 "
                            f"--es msg %s" % charsb64
                        )
                        time.sleep(1)
                        self._d.shell([
                            "ime", "disable",
                            'com.android.adbkeyboard/.AdbIME'
                        ])
                    else:
                        self._d.shell(["input", "text", text])

                    time.sleep(len(text) * 0.1)

                case 'key':
                    text = action.parameters['text']
                    self._d.keyevent(text)

                case 'swipe' | 'scroll':
                    x1, y1 = action.parameters.get('start_point') or action.parameters.get('coordinate')
                    x2, y2 = action.parameters.get('end_point') or action.parameters.get('coordinate2')
                    self._d.swipe(x1, y1, x2, y2, duration=0.5)

                case 'press_home':
                    self._d.keyevent("HOME")

                case 'press_back':
                    self._d.keyevent("BACK")

                case 'wait':
                    duration = action.parameters.get('time', 5.0)
                    time.sleep(duration)

                case 'answer':
                    answer = action.parameters['text']
                    os.system(
                        f'adb -P {self.port} '
                        f'-s {self._d.get_serialno()} '
                        f'shell am broadcast '
                        f'com.example.ACTION_UPDATE_OVERLAY '
                        f'--es task_type_string "Agent answered:" '
                        f'--es goal_string "{answer}"'
                    )
                    return answer

                case 'system_button':
                    button = action.parameters['button']
                    if button == 'Back':
                        self._d.keyevent("BACK")
                    elif button == 'Home':
                        self._d.keyevent("HOME")
                    elif button == 'Menu':
                        self._d.keyevent("MENU")
                    elif button == 'Enter':
                        self._d.keyevent("ENTER")

                case 'clear_text':
                    self._d.shell(["input", "text", " "])
                    self._d.shell([
                        "ime", "enable",
                        'com.android.adbkeyboard/.AdbIME'
                    ])
                    self._d.shell([
                        "ime", "set",
                        'com.android.adbkeyboard/.AdbIME'
                    ])
                    time.sleep(1)
                    os.system(
                        f"adb -P {self.port} "
                        f"-s {self._d.get_serialno()} "
                        f"shell am broadcast -a ADB_CLEAR_TEXT"
                    )
                    time.sleep(1)
                    self._d.shell([
                        "ime", "disable",
                        'com.android.adbkeyboard/.AdbIME'
                    ])

                case 'take_note':
                    note = action.parameters['text']
                    return note

                case 'open_url':
                    self._open_url(action.parameters['text'])

                case 'push_file':
                    self._push_file(
                        action.parameters['text'],
                        action.parameters.get(
                            'destination',
                            '/storage/emulated/0/DCIM'
                        ),
                    )

                case 'install_apk':
                    self._install_apk(action.parameters['text'])

                case 'airplane_mode':
                    self._airplane_mode(action.parameters['text'])

                case 'input_emoticon':
                    self._input_emoticon(action.parameters['text'])

                case 'click_by_text':
                    self._click_by_text(
                        action.parameters['text'],
                        action.parameters.get('index', 0),
                    )

                case 'click_by_id':
                    self._click_by_id(
                        action.parameters['text'],
                        action.parameters.get('index', 0),
                    )

                case 'click_by_description':
                    self._click_by_description(
                        action.parameters['text'],
                        action.parameters.get('index', 0),
                    )

                case 'dump_xml':
                    result = self._dump_xml()

                case 'get_clipboard':
                    result = self._get_clipboard()

                case 'fetch_otp':
                    result = self._fetch_otp(action.parameters['port'])

                case _:
                    raise ValueError(f"Unknown action: {action.name}")

        time.sleep(self.wait_after_action_seconds)
        return str(result)
