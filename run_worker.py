#!/usr/bin/env python3
"""macro1 worker — consume jobs from RabbitMQ and execute on an emulator.

Usage:
    python3 run_worker.py social1
    python3 run_worker.py social2

Each worker listens on its own RabbitMQ queue and runs jobs on its mapped emulator.
Designed to run as a systemd service: macro1@social1.service
"""

import json
import logging
import os
import signal
import sys
import time
import traceback

import pika
import requests
from dotenv import load_dotenv

from macro1 import Agent
from macro1.schema.schema import Action, AgentStatus

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s [%(name)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('worker')

# ── Device mapping ────────────────────────────────────────────────────────────

DEVICE_MAP = {
    'social1': 'emulator-5554',
    'social2': 'emulator-5556',
    'social3': 'emulator-5558',
    'social4': 'emulator-5560',
}

# ── Job → Goal translation ───────────────────────────────────────────────────

def job_to_goal(job: dict) -> str:
    """Translate a job document into a natural language goal for ReActAgent."""
    action = job.get('action', '')
    platform = job.get('platform', '')
    detail = job.get('detail', {})
    url = detail.get('url', '')
    profile_url = detail.get('profile_url', '')
    text = detail.get('text', '')
    comment_text = detail.get('comment', '')

    if action == 'like':
        return f"Open this {platform} post URL and like it: {url}"

    elif action == 'comment':
        comment = comment_text or text
        return f"Open this {platform} post URL and write this comment '{comment}': {url}"

    elif action == 'follow':
        target = profile_url or url
        if platform == 'youtube':
            return f"Open this YouTube channel URL and subscribe: {target}"
        return f"Open this {platform} profile URL and follow: {target}"

    elif action == 'posting':
        caption = text or ''
        return f"Open {platform}, create a new post with the most recent photo in gallery, caption: {caption}"

    elif action == 'repost':
        if platform in ('twitter', 'x'):
            return f"Open this tweet URL and repost/retweet it: {url}"
        return f"Open this {platform} post URL and share/repost it: {url}"

    elif action == 'view':
        return f"Open this {platform} post URL and view it: {url}"

    else:
        # Fallback: describe what we know
        return f"On {platform}, perform '{action}' on: {url or profile_url}"


# ── Job status API ────────────────────────────────────────────────────────────

def update_jobstatus(jobid: str, status: str, posturl: str = "",
                     detail: str = None, error_detail: str = None,
                     exec_time: float = 0):
    """Update job status via REST API."""
    api_url = os.getenv('JOB_STATUS_API')
    if not api_url:
        logger.warning("JOB_STATUS_API not set, skipping status update")
        return False

    data = {
        "status": status,
        "detail": {
            "posturl": posturl,
            "error": detail,
            "error_detail": error_detail,
            "exec_time": exec_time / 60
        }
    }
    try:
        r = requests.patch(f"{api_url}{jobid}", json=data, timeout=10)
        if r.status_code != 200:
            logger.warning(f"update_jobstatus got {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        logger.error(f"update_jobstatus failed: {e}")
        return False


def create_job_log(jobid: str, status: str, posturl: str = "",
                   detail: str = None):
    """Create job log entry via REST API."""
    api_url = os.getenv('JOB_LOG_API')
    if not api_url:
        logger.warning("JOB_LOG_API not set, skipping log creation")
        return False

    log_data = {
        "jobid": jobid,
        "status": status,
        "detail": {
            "error": detail,
            "posturl": posturl,
        }
    }
    try:
        r = requests.post(api_url, json=log_data, timeout=10)
        return r.status_code == 200
    except Exception as e:
        logger.error(f"create_job_log failed: {e}")
        return False


# ── Worker ────────────────────────────────────────────────────────────────────

class Worker:
    def __init__(self, queue_name: str):
        self.queue_name = queue_name
        self.serial_no = DEVICE_MAP.get(queue_name)
        if not self.serial_no:
            raise ValueError(f"Unknown queue '{queue_name}'. Valid: {list(DEVICE_MAP.keys())}")

        self.current_jobid = None
        self.agent = None
        self._connection = None
        self._channel = None

        # Build agent once — reused across jobs
        logger.info(f"Initializing agent for {queue_name} ({self.serial_no})")
        self.agent = Agent.from_params({
            "type": "SingleAgent",
            "env": {
                "serial_no": self.serial_no,
                "host": "127.0.0.1",
                "port": 5037,
                "go_home": False,
            },
            "vlm": {
                "model_name": os.getenv("VLM_MODEL", "qwen/qwen3.5-397b-a17b"),
                "api_key": os.environ["VLM_API_KEY"],
                "base_url": os.environ["VLM_BASE_URL"],
            },
            "max_steps": int(os.getenv("MAX_STEPS", "20")),
        })
        logger.info(f"Agent ready for {queue_name}")

    def connect_rabbitmq(self):
        """Connect to RabbitMQ and declare the queue."""
        credentials = pika.PlainCredentials(
            os.getenv('RABBIT_USER', 'user'),
            os.getenv('RABBIT_PASS', 'pass'),
        )
        params = pika.ConnectionParameters(
            host=os.getenv('RABBIT_HOST', '10.10.0.44'),
            port=int(os.getenv('RABBIT_PORT', '5673')),
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=300,
        )
        self._connection = pika.BlockingConnection(params)
        self._channel = self._connection.channel()
        self._channel.queue_declare(queue=self.queue_name, durable=True)
        self._channel.basic_qos(prefetch_count=1)
        logger.info(f"Connected to RabbitMQ, queue: {self.queue_name}")

    def on_job(self, ch, method, properties, body):
        """Callback for each job message from RabbitMQ."""
        start_time = time.time()
        job = None
        jobid = None

        try:
            job = json.loads(body.decode())
            jobid = job.get('_id', {}).get('$oid', 'unknown')
            self.current_jobid = jobid
            action = job.get('action', '?')
            platform = job.get('platform', '?')

            logger.info(f"Job received: {jobid} [{action} on {platform}]")

            # Acknowledge immediately so RabbitMQ doesn't redeliver
            ch.basic_ack(delivery_tag=method.delivery_tag)

            # Random delay if specified
            random_wait = job.get('detail', {}).get('random', 0)
            if random_wait:
                logger.info(f"Waiting {random_wait}s (random delay)")
                time.sleep(random_wait)

            # Mark as In Progress
            update_jobstatus(jobid, "In Progress")

            # Always start from home screen and close all running apps
            self.agent.env.execute_action(Action(name='press_home', parameters={}))
            time.sleep(1)
            self.agent.env._u2.app_stop_all()
            time.sleep(1)

            # Translate job → goal
            goal = job_to_goal(job)
            logger.info(f"Goal: {goal}")

            # Run agent
            for i, step in enumerate(self.agent.iter_run(goal)):
                if step is None:
                    break
                action_data = step.action
                if action_data:
                    logger.info(f"  Step {i}: [{action_data.name}] {action_data.parameters}")

            # Check result
            ep = self.agent.episode_data
            exec_time = time.time() - start_time

            if ep.status == AgentStatus.FINISHED:
                logger.info(f"Job {jobid} completed in {exec_time:.1f}s")
                update_jobstatus(jobid, "OK", exec_time=exec_time)
                create_job_log(jobid, "OK")
                # Show success on device
                success_url = os.getenv('JOB_SUCCESS_URL')
                if success_url:
                    self.agent.env.execute_action(
                        Action(name='open_url', parameters={'text': success_url})
                    )
            else:
                error_msg = ep.message or "Agent did not finish"
                logger.warning(f"Job {jobid} failed: {error_msg}")
                update_jobstatus(jobid, "Error", detail=error_msg, exec_time=exec_time)
                create_job_log(jobid, "Error", detail=error_msg)
                # Show failure on device
                failed_url = os.getenv('JOB_FAILED_URL')
                if failed_url:
                    self.agent.env.execute_action(
                        Action(name='open_url', parameters={'text': failed_url})
                    )

        except Exception as e:
            exec_time = time.time() - start_time
            error_msg = f"Worker error: {str(e)}"
            logger.error(f"Job {jobid} exception: {error_msg}")
            logger.error(traceback.format_exc())

            if jobid:
                update_jobstatus(jobid, "Error", detail=error_msg,
                                 error_detail=traceback.format_exc(),
                                 exec_time=exec_time)
                create_job_log(jobid, "Error", detail=error_msg)

            # Try to nack if we haven't acked yet
            try:
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            except Exception:
                pass

        finally:
            self.current_jobid = None

    def handle_shutdown(self, signum, frame):
        """Graceful shutdown: mark in-progress job as Error."""
        logger.info(f"Shutdown signal received ({signum})")
        if self.current_jobid:
            logger.info(f"Marking job {self.current_jobid} as Error (service stopped)")
            update_jobstatus(self.current_jobid, "Error", detail="Service stopped")
            create_job_log(self.current_jobid, "Error", detail="Service stopped")
        if self._connection and self._connection.is_open:
            self._connection.close()
        sys.exit(0)

    def run(self):
        """Start consuming jobs. Reconnects on connection loss."""
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        signal.signal(signal.SIGINT, self.handle_shutdown)

        while True:
            try:
                self.connect_rabbitmq()
                self._channel.basic_consume(
                    queue=self.queue_name,
                    on_message_callback=self.on_job,
                )
                logger.info(f"Waiting for jobs on queue '{self.queue_name}'...")
                self._channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as e:
                logger.error(f"RabbitMQ connection lost: {e}. Reconnecting in 5s...")
                time.sleep(5)
            except Exception as e:
                logger.error(f"Unexpected error: {e}. Restarting in 5s...")
                logger.error(traceback.format_exc())
                time.sleep(5)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <queue_name>")
        print(f"  e.g. python3 {sys.argv[0]} social1")
        print(f"  Valid queues: {list(DEVICE_MAP.keys())}")
        sys.exit(1)

    queue_name = sys.argv[1]
    worker = Worker(queue_name)
    worker.run()


if __name__ == '__main__':
    main()
