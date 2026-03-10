
import logging

logger = logging.getLogger(__name__)



ACTION_SPACE = {
    "click": {
        "name": "CLICK",
        "description": "Click on the screen at the given position.",
        "parameters": {
            "point": {
                "type": "array",
                "description": "The coordinate point position to click, such as [230, 560]."
            }
        }
    },
    "long_press": {
        "name": "LONG PRESS",
        "description": "Long press on the screen at the given position.",
        "parameters": {
            "point": {
                "type": "array",
                "description": "The coordinate point position to long press, such as [230, 560]."
            }
        }
    },
    "type": {
        "name": "TYPE",
        "description": "Type text on the screen.",
        "parameters": {
            "text": {"type": "str", "description": "The text to type."}
        }
    },
    "scroll": {
        "name": "SCROLL",
        "description": "Scrolls at (x, y) in the given direction.",
        "parameters": {
            "start_point": {
                "type": "array",
                "description": "The coordinate point position to the scroll start point, such as [230, 560]."
            },
            "end_point": {
                "type": "array",
                "description": "The coordinate point position to the scroll end point, such as [230, 560]."
            }
        }
    },
    "press_home": {
        "name": "PRESS HOME",
        "description": "Press the home button."
    },
    "press_back": {
        "name": "PRESS BACK",
        "description": "Press the back button."
    },
    "wait": {
        "name": "WAIT",
        "description": "Wait for a brief moment."
    },
    "finished": {
        "name": "FINISHED",
        "description": "A special flag indicating that the task has been completed",
        "parameters": {
            "answer": {
                "type": "str",
                "description": "The final answer for the task goal."
            }
        }
    },
    "open_app": {
        "name": "OPEN APP",
        "description": "Open an app by name using adb (no need to find icon coordinates). Supported: instagram, whatsapp, telegram, chrome, youtube, tiktok, facebook, twitter, snapchat, discord, spotify, settings, camera, gmail, google maps, play store, and more.",
        "parameters": {
            "text": {"type": "str", "description": "The app name to open, e.g. 'instagram', 'whatsapp', 'chrome'."}
        }
    },
    "fetch_otp": {
        "name": "FETCH OTP",
        "description": "Fetch the latest OTP code from the SIM card SMS inbox via modem API.",
        "parameters": {
            "port": {"type": "str", "description": "The modem proxy port, e.g. '8081', '8082', '8083', '8084'."}
        }
    },
    "call_user": {
        "name": "ASK HUMAN",
        "description": "Ask human for help."
    },
    "click_by_text": {
        "name": "CLICK BY TEXT",
        "description": "Click on a UI element by its visible text label.",
        "parameters": {
            "text": {"type": "str", "description": "The visible text on the element to click."},
            "index": {"type": "int", "description": "Which instance to click if multiple matches (0-based, default 0)."}
        }
    },
    "click_by_id": {
        "name": "CLICK BY ID",
        "description": "Click on a UI element by its resource ID.",
        "parameters": {
            "text": {"type": "str", "description": "The resource ID, e.g. 'com.instagram.android:id/button_login'."},
            "index": {"type": "int", "description": "Which instance to click if multiple matches (0-based, default 0)."}
        }
    },
    "click_by_description": {
        "name": "CLICK BY DESCRIPTION",
        "description": "Click on a UI element by its content description (accessibility label).",
        "parameters": {
            "text": {"type": "str", "description": "The content description of the element."},
            "index": {"type": "int", "description": "Which instance to click if multiple matches (0-based, default 0)."}
        }
    },
    "dump_xml": {
        "name": "DUMP XML",
        "description": "Dump the current UI hierarchy as XML. Use this to inspect element IDs, text, and descriptions when the screenshot is unclear."
    },
    "get_clipboard": {
        "name": "GET CLIPBOARD",
        "description": "Get the current clipboard content."
    },
    "key": {
        "name": "KEY",
        "description": "Press a key event (e.g. ENTER, DELETE, TAB).",
        "parameters": {
            "text": {"type": "str", "description": "The key to press, e.g. 'ENTER', 'DELETE', 'TAB', 'BACK'."}
        }
    },
    "clear_text": {
        "name": "CLEAR TEXT",
        "description": "Clear the text in the currently focused input field."
    },
    "open_url": {
        "name": "OPEN URL",
        "description": "Open a URL in the device browser.",
        "parameters": {
            "text": {"type": "str", "description": "The URL to open, e.g. 'https://google.com'."}
        }
    },
    "input_emoticon": {
        "name": "INPUT EMOTICON",
        "description": "Input emoji or special characters that cannot be typed normally.",
        "parameters": {
            "text": {"type": "str", "description": "The emoji or special text to input."}
        }
    },
    "airplane_mode": {
        "name": "AIRPLANE MODE",
        "description": "Toggle airplane mode on or off.",
        "parameters": {
            "text": {"type": "str", "description": "'on' or 'off'."}
        }
    }
}
