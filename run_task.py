#!/usr/bin/env python3
"""mobile-use CLI — run a task on an Android emulator"""

import argparse
import logging
import sys
from mobile_use import Environment, VLMWrapper, Agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('run_task')

OPENROUTER_KEY = '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'
OPENROUTER_URL = 'https://openrouter.ai/api/v1'

def main():
    parser = argparse.ArgumentParser(description='Run mobile-use task on Android emulator')
    parser.add_argument('task', help='Task to execute, e.g. "Open YouTube and search for cats"')
    parser.add_argument('--device', '-d', default='emulator-5554',
                        help='ADB device serial (default: emulator-5554)')
    parser.add_argument('--host', default='127.0.0.1', help='ADB host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5037, help='ADB port (default: 5037)')
    parser.add_argument('--model', '-m', default='qwen/qwen3-vl-235b-a22b-thinking',
                        help='VLM model (default: openai/gpt-5.3-codex)')
    parser.add_argument('--max-steps', type=int, default=20, help='Max steps (default: 20)')
    parser.add_argument('--no-home', action='store_true', help='Skip reset to HOME screen')
    args = parser.parse_args()

    logger.info(f'Device  : {args.device} @ {args.host}:{args.port}')
    logger.info(f'Model   : {args.model}')
    logger.info(f'Task    : {args.task}')
    logger.info(f'Steps   : max {args.max_steps}')
    print()

    # Setup environment
    env = Environment(
        serial_no=args.device,
        host=args.host,
        port=args.port,
        go_home=not args.no_home,
    )

    # Setup VLM
    vlm = VLMWrapper(
        model_name=args.model,
        api_key=OPENROUTER_KEY,
        base_url=OPENROUTER_URL,
    )

    # Setup agent
    agent = Agent.from_params({
        "type": "SingleAgent",
        'env': env,
        'vlm': vlm,
        'max_steps': args.max_steps,
    })

    # Run
    logger.info('Starting agent...')
    for i, step in enumerate(agent.iter_run(args.task)):
        if step is None:
            break
        action = step.action
        if action:
            logger.info(f'Step {i+1}: [{action.name}] {action.parameters}')

    logger.info('Done!')

if __name__ == '__main__':
    main()
