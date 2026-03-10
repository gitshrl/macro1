"""Generic Android environment using adbutils + uiautomator2."""

import json
import logging
import time
import traceback
import xml.etree.ElementTree as ET
from typing import Optional

import adbutils
import uiautomator2 as u2
from macro1.schema.schema import Action, EnvState

logger = logging.getLogger(__name__)

# ── App name → package mapping ──────────────────────────────────────────────

APP_PACKAGES = {
    "instagram": "com.instagram.android",
    "facebook": "com.facebook.katana",
    "tiktok": "com.zhiliaoapp.musically",
    "youtube": "com.google.android.youtube",
    "twitter": "com.twitter.android",
    "x": "com.twitter.android",
    "whatsapp": "com.whatsapp",
}

SELECTOR_TIMEOUT = 5


class Environment:
    def __init__(
        self,
        serial_no: str = None,
        host: str = "127.0.0.1",
        port: int = 5037,
        go_home: bool = False,
        wait_after_action_seconds: float = 2.0,
    ):
        self.host = host
        self.port = port
        self.serial_no = serial_no
        self.wait_after_action_seconds = wait_after_action_seconds
        self._register_function = {}

        # Connect via adbutils (for screenshot, shell, install)
        try:
            adb = adbutils.AdbClient(host=host, port=port)
            self._d = adb.device(serial_no)
        except Exception as e:
            logger.error(f"Failed to connect adb to device: {serial_no}")
            raise e

        # Connect via uiautomator2 (for UI interaction)
        try:
            self._u2 = u2.connect(serial_no)
        except Exception as e:
            logger.error(f"Failed to connect u2 to device: {serial_no}")
            raise e

        self.window_size = self._d.window_size(landscape=False)
        if go_home:
            self._u2.press("home")

    # ── State ────────────────────────────────────────────────────────────────

    def get_state(self, display_id: int = -1) -> EnvState:
        try:
            pixels = self._d.screenshot(display_id, error_ok=False)
        except Exception as e:
            raise ValueError(f"Screenshot error: {traceback.format_exc()}") from e
        package = self._d.app_current().package
        device_time = self._d.shell("date")
        return EnvState(pixels=pixels, package=package, device_time=device_time)

    # ── Custom action registration ───────────────────────────────────────────

    def register_action(self, action_name: str, action_func):
        if not callable(action_func):
            raise ValueError(f"Action function for {action_name} must be callable.")
        self._register_function[action_name] = action_func

    # ── Accessibility tree (android-action-kernel pattern) ───────────────────

    def _get_ui_elements(self) -> str:
        """Parse UI hierarchy into structured JSON of interactive elements."""
        xml = self._u2.dump_hierarchy()
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            logger.warning("Failed to parse UI XML")
            return json.dumps([])

        elements = []
        for node in root.iter("node"):
            is_clickable = node.attrib.get("clickable") == "true"
            class_name = node.attrib.get("class", "")
            is_editable = (
                "EditText" in class_name
                or "AutoCompleteTextView" in class_name
                or node.attrib.get("editable") == "true"
            )
            text = node.attrib.get("text", "")
            desc = node.attrib.get("content-desc", "")
            res_id = node.attrib.get("resource-id", "")

            if not is_clickable and not is_editable and not text and not desc:
                continue

            bounds = node.attrib.get("bounds", "")
            if not bounds:
                continue
            try:
                coords = bounds.replace("][", ",").replace("[", "").replace("]", "").split(",")
                x1, y1, x2, y2 = map(int, coords)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            except Exception:
                continue

            if is_editable:
                action = "type"
            elif is_clickable:
                action = "tap"
            else:
                action = "read"

            elements.append({
                "id": res_id,
                "text": text or desc,
                "type": class_name.rsplit(".", 1)[-1],
                "center": [cx, cy],
                "clickable": is_clickable,
                "editable": is_editable,
                "action": action,
            })

        result = json.dumps(elements, ensure_ascii=False)
        logger.info(f"get_ui_elements: {len(elements)} elements found")
        return result

    # ── Action dispatch ──────────────────────────────────────────────────────

    def execute_action(self, action: Action) -> Optional[str]:
        result = None

        if action.name in self._register_function:
            result = self._register_function[action.name](self, **action.parameters)
            time.sleep(self.wait_after_action_seconds)
            return str(result)

        match action.name:

            # -- App management --

            case "open_app":
                app_name = action.parameters["text"].strip().lower()
                package = APP_PACKAGES.get(app_name)
                if package:
                    logger.info(f"Opening app: {app_name} -> {package}")
                    self._u2.app_start(package)
                else:
                    logger.warning(f"Unknown app '{app_name}', trying as package name")
                    self._u2.app_start(app_name)

            case "open_url":
                url = action.parameters["text"]
                self._d.shell(["am", "start", "-a", "android.intent.action.VIEW", "-d", url])

            # -- Coordinate-based interactions --

            case "click":
                x, y = action.parameters.get("point") or action.parameters.get("coordinate")
                self._u2.click(x, y)

            case "long_press":
                x, y = action.parameters.get("point") or action.parameters.get("coordinate")
                duration = action.parameters.get("time", 2.0)
                self._u2.long_click(x, y, duration=duration)

            case "swipe" | "scroll":
                # Direction-based: scroll(direction='up')
                direction = action.parameters.get("direction")
                if direction:
                    self._u2.swipe_ext(direction, scale=0.8)
                else:
                    # Coordinate-based: scroll(start_point=[x1,y1], end_point=[x2,y2])
                    x1, y1 = action.parameters.get("start_point") or action.parameters.get("coordinate")
                    x2, y2 = action.parameters.get("end_point") or action.parameters.get("coordinate2")
                    self._u2.swipe(x1, y1, x2, y2, duration=0.5)

            # -- Text input --

            case "type":
                text = action.parameters["text"]
                self._u2.send_keys(text)

            case "clear_text":
                try:
                    self._u2.clear_text()
                except Exception:
                    # Fallback: select all + delete
                    self._u2.keyevent("KEYCODE_MOVE_HOME")
                    self._u2.keyevent("KEYCODE_MOVE_END", metastate=1)  # shift+end
                    self._u2.keyevent("KEYCODE_DEL")

            case "key":
                key = action.parameters["text"]
                self._u2.press(key.lower())

            case "input_emoticon":
                text = action.parameters["text"]
                # Use u2 clipboard to paste emoji
                self._u2.set_clipboard(text)
                self._u2.keyevent("PASTE")

            # -- Navigation --

            case "press_home":
                self._u2.press("home")

            case "press_back":
                self._u2.press("back")

            case "wait":
                duration = action.parameters.get("time", 5.0)
                time.sleep(duration)

            # -- Element-based interactions (uiautomator2) --

            case "click_by_text":
                text = action.parameters["text"]
                index = action.parameters.get("index", 0)
                el = self._u2(text=text, instance=index)
                if el.click_exists(timeout=SELECTOR_TIMEOUT):
                    logger.info(f"Clicked element by text: {text}")
                else:
                    # Fallback: try partial match
                    el2 = self._u2(textContains=text, instance=index)
                    if el2.click_exists(timeout=SELECTOR_TIMEOUT):
                        logger.info(f"Clicked element by textContains: {text}")
                    else:
                        logger.warning(f"Text element not found: {text}")

            case "click_by_id":
                res_id = action.parameters["text"]
                index = action.parameters.get("index", 0)
                el = self._u2(resourceId=res_id, instance=index)
                if el.click_exists(timeout=SELECTOR_TIMEOUT):
                    logger.info(f"Clicked element by ID: {res_id}")
                else:
                    logger.warning(f"Resource ID not found: {res_id}")

            case "click_by_description":
                desc = action.parameters["text"]
                index = action.parameters.get("index", 0)
                el = self._u2(description=desc, instance=index)
                if el.click_exists(timeout=SELECTOR_TIMEOUT):
                    logger.info(f"Clicked element by desc: {desc}")
                else:
                    el2 = self._u2(descriptionContains=desc, instance=index)
                    if el2.click_exists(timeout=SELECTOR_TIMEOUT):
                        logger.info(f"Clicked element by descContains: {desc}")
                    else:
                        logger.warning(f"Description not found: {desc}")

            # -- Screen analysis --

            case "get_ui_elements":
                result = self._get_ui_elements()

            case "dump_xml":
                result = self._u2.dump_hierarchy()

            # -- Device control --

            case "airplane_mode":
                state = action.parameters["text"]
                if state == "on":
                    self._d.shell(["cmd", "connectivity", "airplane-mode", "enable"])
                elif state == "off":
                    self._d.shell(["cmd", "connectivity", "airplane-mode", "disable"])

            case "open_notification":
                self._u2.open_notification()

            # -- Terminal actions (handled by agent, not executed) --

            case "finished" | "call_user":
                pass

            case _:
                raise ValueError(f"Unknown action: {action.name}")

        time.sleep(self.wait_after_action_seconds)
        return str(result) if result else None
