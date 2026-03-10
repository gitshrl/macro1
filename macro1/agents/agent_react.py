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
You are using a Mobile device. You are able to use a Action Space Operator to interact with the mobile based on the given task and screenshot.

## Action Space
Your available "Next Action" only include:
- open_app(text='instagram'): Directly open an app by name. Supported: instagram, facebook, tiktok, youtube, twitter/x, whatsapp, telegram, chrome, snapchat, discord, spotify, settings, camera, gmail, google maps, play store, messenger, line, pinterest, reddit, linkedin, shopee, tokopedia, gojek, grab, dana, ovo.
- click(point=[x,y]): Click on the coordinate point specified on the screen (x,y).
- long_press(point=[x,y]): Long press the screen to specify coordinates (x,y).
- type(text='hello world'): Types a string of text.
- scroll(start_point=[x1,y1], end_point=[x2,y2]): Scroll the screen, (x1,y1) is the starting coordinate position, (x2,y2) is the end coordinate position. In particular, when y1=y2, you can swipe left and right on the desktop to switch pages, which is very helpful for finding a specific application.
- press_home(): Back to Home page.
- press_back(): Back to previous page.
- finished(answer=''): Submit the task regardless of whether it succeeds or fails. The answer parameter is to summarize the content of the reply to the user.
- call_user(question=''): Submit the task and call the user when the task is unsolvable, or when you need the user's help.
- wait(): Wait for loading to complete.
- fetch_otp(port='8081'): Fetch the latest OTP code from the SIM card SMS inbox. Use the port matching the device: social1='8081', social2='8082', social3='8083', social4='8084'.
- click_by_text(text='Login', index=0): Click on a UI element by its visible text. Useful when you can read text on screen but coordinates are hard to determine. index is optional (default 0).
- click_by_id(text='com.app:id/btn_login', index=0): Click on a UI element by its resource ID. Use dump_xml first to find the ID. index is optional (default 0).
- click_by_description(text='Search', index=0): Click on a UI element by its content description (accessibility label). index is optional (default 0).
- dump_xml(): Dump the current UI hierarchy as XML. Use this when the screenshot is unclear to inspect element text, IDs, and descriptions.
- get_clipboard(): Get the current clipboard content.
- key(text='ENTER'): Press a key event. Common keys: ENTER, DELETE, TAB, BACK, MENU.
- clear_text(): Clear the text in the currently focused input field.
- open_url(text='https://google.com'): Open a URL directly in the device browser.
- input_emoticon(text='😀'): Input emoji or special characters that cannot be typed normally.
- airplane_mode(text='on'): Toggle airplane mode. Use 'on' or 'off'.

## Note
- Action click, long_press and scroll must contain coordinates within.
- You may be given some history plan and actions, this is the response from the previous loop.
- You should carefully consider your plan base on the task, screenshot, and history actions.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## Suggestions
- **ALWAYS use open_app(text='appname') to open apps directly.** Do NOT manually search for app icons on the home screen or open Play Store to find apps. open_app is instant and saves many steps.
- When the screen of the previous operation is not responsive, you need to avoid performing the same action in the next step.
- Shopping or life services apps, you should make use of the in-app search function as much as possible to find quickly.
- Reduce the execution steps as much as possible, and find the optimal execution path to achieve the task goal.

## Format
Task: The task description.
Observation: The mobile screenshot or user response.
Thought: The process of thinking.
Action: The next action. Must use the exact function call syntax from Action Space.

**IMPORTANT: You MUST output the Action using the exact function call syntax with actual coordinates from the screenshot. Do NOT describe the action in natural language.**

Example output:
Thought: I need to open Instagram. I'll use open_app to launch it directly.
Action: open_app(text='instagram')

**Be aware that Observation, Thought, and Action will be repeated.**

Now, let's begin!
""".strip()




IMAGE_PLACEHOLDER = '<|vision_start|><|image_pad|><|vision_end|>'


def parse_reason_and_action(content: str, size: tuple[float, float], raw_size: tuple[float, float]) -> Action:
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
                reason, action, action_r = parse_reason_and_action(content, pixels.size, env_state.pixels.size)
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
