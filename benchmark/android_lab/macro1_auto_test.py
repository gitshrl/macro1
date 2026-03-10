import copy
import time
import logging
import traceback
from dataclasses import asdict
import macro1
from third_party.android_lab.evaluation.evaluation import AutoTask
from third_party.android_lab.evaluation.auto_test import AutoTest
from benchmark.android_lab.macro1_executor import AndroidLabEnvironment, Macro1Executor


logger = logging.getLogger(__name__)


class Macro1Agent:
    def __init__(self, agent_config):
        self.agent_config = agent_config

    def construct(self, instruction, controller, config, page_executor):
        agent = macro1.Agent.from_params(params=copy.deepcopy(self.agent_config))
        agent.env = AndroidLabEnvironment(controller, config, page_executor)
        agent.reset(goal=instruction)
        return agent


class Macro1_AutoTask(AutoTask):
    def set_system_prompt(self, instruction):
        pass

    def run_step(self, round_count):
        if self.record.turn_number == 0:
            time.sleep(5)
        self.record.update_before(controller=self.controller, need_screenshot=True, ac_status=self.accessibility)
        step_data = None
        try:
            self.agent.curr_step_idx = self.record.turn_number
            step_data = self.agent.step()
            self.agent.episode_data.num_steps = len(self.agent.trajectory)
            if self.agent.status == macro1.schema.AgentStatus.FINISHED:
                self.page_executor.is_finish = True
                message = "Task completed."
                for step_data in self.agent.trajectory[::-1]:
                    if step_data.answer is not None:
                        message = step_data.answer
                        break
                self.page_executor.current_return = {"operation": "finish", "action": 'finish', "kwargs": {"message": message}}
            step_data_copy = copy.deepcopy(self.agent.trajectory[-1])
            step_data_copy.curr_env_state = None
            step_data_copy.exec_env_state = None
            rsp = asdict(step_data_copy)
        except Exception as e:
            logger.error("Some error happened during the Macro1 agent run.")
            traceback.print_exc()
            rsp = str(e)
        exe_res = self.page_executor('')
        self.record.update_after(exe_res, rsp)
        self.record.turn_number += 1


class Macro1_AutoTest(AutoTest):
    def get_agent(self):
        agent = self.llm_agent.construct(self.instruction, self.controller, self.config, self.page_executor)
        task_agent = Macro1_AutoTask(
            self.instruction,
            self.controller,
            self.page_executor,
            agent,
            self.record,
            self.command_per_step)
        return task_agent

    def get_executor(self):
        return Macro1Executor(self.controller, self.config)
