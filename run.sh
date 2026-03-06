#!/bin/bash
source ~/mobile-use/venv/bin/activate
export DISPLAY=:0
cd ~/mobile-use
python3 -c "
from mobile_use.webui import build_agent_ui_demo
demo = build_agent_ui_demo()
demo.launch(server_name='0.0.0.0', server_port=7860)
"
