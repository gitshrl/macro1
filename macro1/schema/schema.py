from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from PIL import Image


@dataclass(frozen=True)
class EnvState:
    """State of the environment.

    Attributes:
        pixels: Screenshot of the current state.
        package: Current foreground app package name.
        a11y_tree: Current accessibility tree.
        device_time: Current device time in string format, e.g., "Tue Aug 12 03:04:58 GMT 2025".
    """

    pixels: Image.Image
    package: Optional[str] = None
    a11y_tree: Optional[Any] = None
    device_time: Optional[str] = None


@dataclass()
class Action:
    """A structrued representation of an action.
    
    # Example
    result = {'name': 'click', 'parameters': {'x': %d, 'y': %d}}
    action = Action(**result)

    Attributes:
        name: The action type name.
        parameters: The parameters of the action.
    """

    name: str
    parameters: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self._validate()

    def _validate(self):
        pass

    def __repr__(self) -> str:
        kv = []
        if self.parameters:
            for k, v in self.parameters.items():
                if v is not None:
                    kv.append(f"{k}={v}")
        params_str = ','.join(kv)
        return f"{self.name}({params_str})"
    
    def __str__(self) -> str:
        return self.__repr__()


class AgentState(Enum):
    READY = 'READY'
    RUNNING = 'RUNNING'
    CALLUSER = 'CALLUSER'

class AgentStatus(Enum):
    FINISHED = 'FINISHED'
    FAILED = 'FAILED'


@dataclass
class VLMCallingData:
    messages: List[Dict[str,Any]]
    response: str


@dataclass
class BaseStepData:
    step_idx: int
    curr_env_state: EnvState
    content: Optional[str] = None       # VLM response content
    action: Optional[Action] = None
    exec_env_state: Optional[EnvState] = None
    vlm_call_history: Optional[List[VLMCallingData]] = field(default_factory=list)


@dataclass
class SingleAgentStepData(BaseStepData):
    thought: Optional[str] = None
    action_s: Optional[str] = None
    action_desc: Optional[str] = None
    answer: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class BaseEpisodeData:
    goal: str
    num_steps: Optional[int] = None
    status: Optional[str] = None
    message: Optional[str] = None
    trajectory: Optional[List[BaseStepData]] = None
