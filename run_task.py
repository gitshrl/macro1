#!/usr/bin/env python3
"""macro1 CLI — run a task on an Android emulator"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv
from macro1 import Agent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('run_task')

def main():
    parser = argparse.ArgumentParser(description='Run macro1 task on Android emulator')
    parser.add_argument('task', help='Task to execute, e.g. "Open YouTube and search for cats"')
    parser.add_argument('--device', '-d', default='emulator-5554',
                        help='ADB device serial (default: emulator-5554)')
    parser.add_argument('--host', default='127.0.0.1', help='ADB host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5037, help='ADB port (default: 5037)')
    parser.add_argument('--model', '-m', default='qwen/qwen3.5-397b-a17b',
                        help='VLM model (default: qwen/qwen3.5-397b-a17b)')
    parser.add_argument('--max-steps', type=int, default=20, help='Max steps (default: 20)')
    parser.add_argument('--no-home', action='store_true', help='Skip reset to HOME screen')
    args = parser.parse_args()

    logger.info(f'Device  : {args.device} @ {args.host}:{args.port}')
    logger.info(f'Model   : {args.model}')
    logger.info(f'Task    : {args.task}')
    logger.info(f'Steps   : max {args.max_steps}')
    print()

    # Setup agent
    agent = Agent.from_params({
        "type": "SingleAgent",
        'env': {
            'serial_no': args.device,
            'host': args.host,
            'port': args.port,
            'go_home': not args.no_home,
        },
        'vlm': {
            'model_name': args.model,
            'api_key': os.environ['VLM_API_KEY'],
            'base_url': os.environ['VLM_BASE_URL'],
        },
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

    ep = agent.episode_data
    if ep.message:
        logger.error(f'Agent error: {ep.message}')
    logger.info(f'Status: {ep.status}, Steps: {ep.num_steps}')

if __name__ == '__main__':
    main()
