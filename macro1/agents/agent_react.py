import re
import logging
from typing import Iterator

from macro1.action import ACTION_SPACE
from macro1.schema.schema import *
from macro1.environment.mobile_environ import Environment
from macro1.utils.vlm import VLMWrapper
from macro1.utils.utils import encode_image_url, smart_resize
from macro1.agents import Agent
from macro1.schema.config import ReActAgentConfig


logger = logging.getLogger(__name__)


SYSTEM_PROMPT_EN = """
You are an Android automation agent. You interact with a mobile device using screenshots and an action space.

## Action Space

### App Management
- open_app(text='instagram'): Open an app by name. Supported: instagram, facebook, tiktok, youtube, twitter/x, whatsapp. For other apps, use the package name directly.
- open_url(text='https://google.com'): Open a URL in the browser.

### Touch Interactions
- click(point=[x,y]): Tap a coordinate on the screen.
- long_press(point=[x,y]): Long press a coordinate.
- scroll(direction='up'): Scroll the screen. Direction: 'up', 'down', 'left', 'right'. Or use scroll(start_point=[x1,y1], end_point=[x2,y2]) for precise control.

### Text Input
- type(text='hello'): Type text into the focused input field.
- clear_text(): Clear the focused input field.
- key(text='enter'): Press a key: enter, delete, back, home, menu, search.

### Navigation
- press_home(): Go to the home screen.
- press_back(): Go back to the previous screen.
- wait(): Wait for the screen to load.

### Element-Based Interactions (by UI properties, no coordinates needed)
- click_by_text(text='Login'): Click an element by its visible text. Optional: index=0 for multiple matches.
- click_by_id(text='com.app:id/btn'): Click an element by resource ID. Use get_ui_elements() to find IDs.
- click_by_description(text='Search'): Click an element by its accessibility description.

### Screen Analysis
- get_ui_elements(): Get a structured list of all interactive elements (text, type, center coordinates, IDs). Use when the screenshot is unclear or you need resource IDs.
- dump_xml(): Get the raw UI hierarchy XML.
- get_clipboard(): Read the current clipboard text. Use after copying text or to read auto-copied OTP codes.

### Device Control
- open_notification(): Open the notification panel.

### Task Control
- finished(answer=''): Mark the task as completed with a summary.
- call_user(question=''): Ask the user for help when the task is unsolvable.

## Rules
- Use open_app() to open apps. Do NOT search for app icons on the home screen.
- When the screen doesn't change after an action, try a different approach.
- Use click_by_text() when you can see text labels clearly — it's more reliable than coordinates.
- Use get_ui_elements() when the screenshot is ambiguous — it gives you exact element positions and IDs.
- Minimize steps. Find the most efficient path to complete the task.

## Format
Thought: <your reasoning>
Action: <exact function call>

Example:
Thought: I need to open Instagram to complete the task.
Action: open_app(text='instagram')

Observation, Thought, and Action repeat until the task is done.
""".strip()




IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'


def parse_reason_and_action(content: str, raw_size: tuple[float, float]) -> Action:
    # Strip <think>...</think> blocks from thinking models
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    reason = re.search(r'Thought:(.*)Action:', content, flags=re.DOTALL)
    if reason:
        reason_s = reason.group(1).strip()
    else:
        reason_s = None

    action_name = '|'.join(ACTION_SPACE.keys())
    # Try standard format first: Action: click(...)
    search_res = re.search(fr'Action: *({action_name})\((.*)\)', content, flags=re.DOTALL)

    # Fallback: action call inside markdown code blocks
    if not search_res:
        search_res = re.search(fr'```\s*\n?\s*({action_name})\((.*?)\)\s*\n?\s*```', content, flags=re.DOTALL)

    # Fallback: bare action call anywhere in content
    if not search_res:
        search_res = re.search(fr'({action_name})\((.*?)\)', content, flags=re.DOTALL)

    if not search_res:
        raise Exception("Action is undefined")

    name = search_res.group(1).strip()
    params = eval(f"dict({search_res.group(2)})")

    for k, v in params.items():
        if ACTION_SPACE[name].get('parameters', {}).get(k, {}).get('type') == 'array':
            try:
                # Qwen outputs 0-1000 normalized coords
                x = round(v[0] / 1000 * raw_size[0])
                y = round(v[1] / 1000 * raw_size[1])
                params[k] = (x, y)
            except:
                pass
    action_a = Action(name=name, parameters=params)
    action_r = f'{name}({search_res.group(2)})'     # raw action
    return reason_s, action_a, action_r


@Agent.register('SingleAgent')
@Agent.register('ReAct')
class ReActAgent(Agent):
    def __init__(self, config_path: str = None, **kwargs):
        super().__init__(config_path, **kwargs)
        if config_path is not None:
            self.config = ReActAgentConfig.from_yaml(config_path)
        else:
            self.config = ReActAgentConfig(**kwargs)
        self.num_latest_screenshot = self.config.num_latest_screenshots
        self.max_reflection_action = self.config.max_action_retry

    def reset(self, goal: str='', max_steps: int = None) -> None:
        """Reset the state of the agent."""
        if max_steps is not None:
            self.max_steps = max_steps
        self._init_data(goal=goal)
        self._recent_actions = []
        self._max_repeat = 3

    def _remain_most_recent_images(self):
        couter = 0
        for i in range(len(self.messages)-1, -1, -1):
            message = self.messages[i]
            if isinstance(message['content'], list):
                j = len(message['content']) - 1
                while j >= 0:
                    cnt = message['content'][j]
                    if cnt['type'] == 'image_url':
                        if couter >= self.num_latest_screenshot:
                            message['content'].pop(j)
                            message['content'][j-1]['text'] = message['content'][j-1]['text'].replace(IMAGE_PLACEHOLDER, 'None')
                        else:
                            couter += 1
                    j -= 1

    def _get_curr_step_data(self) -> SingleAgentStepData:
        if len(self.trajectory) > self.curr_step_idx:
            return self.trajectory[self.curr_step_idx]
        else:
            return None

    def step(self) -> SingleAgentStepData:
        """Execute the task with maximum number of steps.

        Returns: SingleAgentStepData
        """
        logger.info("Step %d ... ..." % self.curr_step_idx)

        # Init messages
        if self.curr_step_idx == 0:
            system_prompt = SYSTEM_PROMPT_EN
            logger.info(f"system_prompt:\n{system_prompt}")
            self.messages.append({
                'role': 'system', 
                'content': system_prompt
            })
            self.messages.append({
                'role': 'user', 
                'content': [
                    {
                        'type': 'text',
                        'text': f'Task: {self.goal}'
                    }
                ]
            })
        if self.state == AgentState.CALLUSER:
            observation = self._user_input
            img_msg = None
        else:
            observation = ''

            # Get the current environment screen
            env_state = self.env.get_state()
            pixels = env_state.pixels.copy()
            pixels.thumbnail((1024, 1024))
            h, w = smart_resize(height=pixels.height, width=pixels.width)
            pixels = pixels.resize([w, h])
            img_msg = {
                "type": "image_url",
                "image_url": {"url": encode_image_url(pixels)}
            }
            # Add new step data
            self.trajectory.append(SingleAgentStepData(
                step_idx=self.curr_step_idx,
                curr_env_state=env_state,
                vlm_call_history=[]
            ))
        self.messages[-1]['content'].append({
            'type': 'text',
            'text': f'Observation: {observation}'
        })
        if img_msg:
            self.messages[-1]['content'].append(img_msg)

        step_data = self.trajectory[-1]

        self._remain_most_recent_images()
        
        response = self.vlm.predict(self.messages, stop=['Observation'])

        counter = self.max_reflection_action
        reason, action = None, None
        content = None
        while counter > 0:
            try:
                # Handle thinking models where choices may be None
                msg = None
                if response.choices:
                    msg = response.choices[0].message
                    content = msg.content or ''
                else:
                    content = ''

                # For thinking models: use reasoning content as fallback
                if not content.strip() and msg is not None:
                    extras = msg.model_extra or {}
                    reasoning = (
                        getattr(msg, 'reasoning_content', None)
                        or getattr(msg, 'reasoning', None)
                        or extras.get('reasoning')
                        or extras.get('reasoning_content')
                    )
                    if reasoning:
                        content = reasoning
                        logger.info("Using reasoning content as fallback")
                # Also check response-level model_extra for reasoning
                if not content.strip():
                    resp_extras = response.model_extra or {}
                    reasoning = resp_extras.get('reasoning') or resp_extras.get('reasoning_content')
                    if reasoning:
                        content = reasoning
                        logger.info("Using response-level reasoning as fallback")
                if not content.strip():
                    raise Exception("Empty content from VLM")
                step_data.content = content
                logger.info("Content from VLM:\n%s" % step_data.content)
                step_data.vlm_call_history.append(VLMCallingData(self.messages, response))
                reason, action, action_r = parse_reason_and_action(content, env_state.pixels.size)
                logger.info("REASON: %s" % reason)
                logger.info("ACTION: %s" % str(action))
                self.messages[-1]['content'].append({
                    'type': 'text',
                    'text': f'Thought: {reason}\nAction: {action_r}'
                })
                break
            except Exception as e:
                logger.warning(f"Failed to parse the action from content ({type(e).__name__}: {e}).")
                msg = {
                    'type': 'text', 
                    'text': f"Failed to parse the action from: {content}.Error is {e.args}"
                }
                self.messages[-1]['content'].append(msg)
                self._remain_most_recent_images()
                response = self.vlm.predict(self.messages, stop=['Observation'])
                counter -= 1
        if action is None:
            raise Exception("Action parse error after max retry")

        step_data.action = action
        step_data.thought = reason

        # Stuck detection: warn if same action repeats too many times
        action_key = f"{action.name}({action.parameters})"
        self._recent_actions.append(action_key)
        if len(self._recent_actions) >= self._max_repeat:
            last_n = self._recent_actions[-self._max_repeat:]
            if len(set(last_n)) == 1:
                logger.warning(f"Stuck detected: '{action_key}' repeated {self._max_repeat} times")
                self.messages[-1]['content'].append({
                    'type': 'text',
                    'text': (
                        f"WARNING: You have repeated the exact same action '{action.name}' "
                        f"{self._max_repeat} times in a row. The screen is not changing. "
                        f"You MUST try a completely different approach. Consider: "
                        f"press_back(), scroll to find a different element, or "
                        f"re-read the screen carefully for the correct target. "
                        f"Do NOT repeat the same action again."
                    )
                })

        if action.name.upper() == 'FINISHED':
            logger.info(f"Finished: {action}")
            self.status = AgentStatus.FINISHED
            step_data.answer = action.parameters.get('answer')
        elif action.name.upper() == 'CALL_USER':
            logger.info(f"Call for help from user:{action}")
            self.state = AgentState.CALLUSER
        else:
            logger.info(f"Execute the action: {action}")
            self.env.execute_action(action)
            step_data.exec_env_state = self.env.get_state()

        return step_data

    def iter_run(self, input_content: str) -> Iterator[SingleAgentStepData]:
        """Execute the agent with user input content.

        Returns: Iterator[SingleAgentStepData]
        """

        if self.state == AgentState.READY:
            self.reset(goal=input_content)
            logger.info("Start task: %s, with at most %d steps" % (self.goal, self.max_steps))
        elif self.state == AgentState.CALLUSER:
            self._user_input = input_content      # user answer
            self.state = AgentState.RUNNING       # reset agent state
            logger.info("Continue task: %s, with user input %s" % (self.goal, input_content))
        else:
            raise Exception('Error agent state')

        for step_idx in range(self.curr_step_idx, self.max_steps):
            self.curr_step_idx = step_idx
            try:
                # show current environment
                yield SingleAgentStepData(
                    step_idx=self.curr_step_idx,
                    curr_env_state=self.env.get_state(),
                    vlm_call_history=[]
                )
                self.step()
                yield self._get_curr_step_data()
            except Exception as e:
                self.status = AgentStatus.FAILED
                self.episode_data.status = self.status
                self.episode_data.message = str(e)
                yield self._get_curr_step_data()
                return

            self.episode_data.num_steps = step_idx + 1
            self.episode_data.status = self.status

            if self.status == AgentStatus.FINISHED:
                logger.info("Agent indicates task is done.")
                self.episode_data.message = 'Agent indicates task is done'
                yield self._get_curr_step_data()
                return
            elif self.state == AgentState.CALLUSER:
                logger.info("Agent indicates to ask user for help.")
                yield self._get_curr_step_data()
                return
            else:
                logger.info("Agent indicates one step is done.")
            yield self._get_curr_step_data()
        logger.warning(f"Agent reached max number of steps: {self.max_steps}.")

    def run(self, input_content: str) -> BaseEpisodeData:
        """Execute the agent with user input content.

        Returns: BaseEpisodeData
        """
        for _ in self.iter_run(input_content, stream=False):
            pass
        return self.episode_data
