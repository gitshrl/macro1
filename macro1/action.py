"""Action space definition — all actions the VLM model can invoke."""

ACTION_SPACE = {
    # -- App management --
    "open_app": {
        "name": "OPEN APP",
        "description": "Open an app by name.",
        "parameters": {
            "text": {"type": "str", "description": "App name, e.g. 'instagram'."}
        }
    },
    "open_url": {
        "name": "OPEN URL",
        "description": "Open a URL in the device browser.",
        "parameters": {
            "text": {"type": "str", "description": "The URL to open."}
        }
    },

    # -- Coordinate-based interactions --
    "click": {
        "name": "CLICK",
        "description": "Tap a point on the screen.",
        "parameters": {
            "point": {"type": "array", "description": "Coordinate [x, y]."}
        }
    },
    "long_press": {
        "name": "LONG PRESS",
        "description": "Long press a point on the screen.",
        "parameters": {
            "point": {"type": "array", "description": "Coordinate [x, y]."}
        }
    },
    "scroll": {
        "name": "SCROLL",
        "description": "Scroll the screen. Use direction OR start/end points.",
        "parameters": {
            "direction": {"type": "str", "description": "'up', 'down', 'left', or 'right'."},
            "start_point": {"type": "array", "description": "Start coordinate [x, y]."},
            "end_point": {"type": "array", "description": "End coordinate [x, y]."}
        }
    },

    # -- Text input --
    "type": {
        "name": "TYPE",
        "description": "Type text into the focused input field.",
        "parameters": {
            "text": {"type": "str", "description": "The text to type."}
        }
    },
    "clear_text": {
        "name": "CLEAR TEXT",
        "description": "Clear text in the currently focused input field."
    },
    "key": {
        "name": "KEY",
        "description": "Press a key.",
        "parameters": {
            "text": {"type": "str", "description": "Key name: enter, delete, back, home, menu, search, etc."}
        }
    },
    # -- Navigation --
    "press_home": {
        "name": "PRESS HOME",
        "description": "Go to the home screen."
    },
    "press_back": {
        "name": "PRESS BACK",
        "description": "Go back to the previous screen."
    },
    "wait": {
        "name": "WAIT",
        "description": "Wait for the screen to load."
    },

    # -- Element-based interactions (uiautomator2) --
    "click_by_text": {
        "name": "CLICK BY TEXT",
        "description": "Click a UI element by its visible text.",
        "parameters": {
            "text": {"type": "str", "description": "The visible text on the element."},
            "index": {"type": "int", "description": "Instance index if multiple matches (default 0)."}
        }
    },
    "click_by_id": {
        "name": "CLICK BY ID",
        "description": "Click a UI element by its resource ID.",
        "parameters": {
            "text": {"type": "str", "description": "Resource ID, e.g. 'com.app:id/btn'."},
            "index": {"type": "int", "description": "Instance index if multiple matches (default 0)."}
        }
    },
    "click_by_description": {
        "name": "CLICK BY DESCRIPTION",
        "description": "Click a UI element by its accessibility description.",
        "parameters": {
            "text": {"type": "str", "description": "The content description."},
            "index": {"type": "int", "description": "Instance index if multiple matches (default 0)."}
        }
    },

    # -- Screen analysis --
    "get_ui_elements": {
        "name": "GET UI ELEMENTS",
        "description": "Get structured list of interactive UI elements with text, type, center coordinates, and suggested action. Use this when the screenshot is unclear.",
    },
    "dump_xml": {
        "name": "DUMP XML",
        "description": "Dump the raw UI hierarchy as XML. Use get_ui_elements for a cleaner view.",
    },
    "get_clipboard": {
        "name": "GET CLIPBOARD",
        "description": "Read the current clipboard text. Use after long-pressing and copying text, or to read OTP codes that were auto-copied.",
    },

    # -- Device control --
    "open_notification": {
        "name": "OPEN NOTIFICATION",
        "description": "Open the notification panel.",
    },

    # -- Terminal actions --
    "finished": {
        "name": "FINISHED",
        "description": "Mark the task as completed.",
        "parameters": {
            "answer": {"type": "str", "description": "Summary of what was done."}
        }
    },
    "call_user": {
        "name": "ASK HUMAN",
        "description": "Ask the user for help when the task is unsolvable.",
        "parameters": {
            "question": {"type": "str", "description": "The question to ask."}
        }
    },
}
