#!/bin/bash
source ~/macro1/venv/bin/activate
export DISPLAY=:0
cd ~/macro1
python3 -c "
from macro1.webui import build_agent_ui_demo
demo = build_agent_ui_demo()
demo.launch(server_name='0.0.0.0', server_port=7860)
"
