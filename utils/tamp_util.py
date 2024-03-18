from typing import List, Dict, Any, Optional, Dict
from dataclasses import dataclass, field

import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class TAMPFeedback:
    motion_planner_feedback: str = ""
    task_process_feedback: str = ""
    action_success: bool = False
    goal_achieved: bool = False


@dataclass
class Parameter:
    name: str
    lower_limit: float
    upper_limit: float


@dataclass
class Action:
    primitive: "PrimitiveAction"
    obj_args: list = field(default_factory=list)
    param_args: dict = field(default_factory=dict)
    traj: list = field(default_factory=list)

    def __str__(self):
        return f"{self.primitive.name}({self.obj_args}, {self.param_args})"


@dataclass
class PrimitiveAction:
    name: str
    obj_arity: int = 0
    parameters: dict = field(default_factory=dict)

    def __call__(
        self, obj_args: List[str], param_args: Dict[str, float] = {}, traj: List[Any] = []
    ):
        assert len(obj_args) == self.obj_arity, f"Expected {self.obj_arity} object arguments!"
        assert len(param_args) == len(
            self.parameters
        ), f"Expected {len(self.param_names)} param arguments!"

        for param in param_args.keys():
            if param not in self.parameters:
                param_args.pop(param)
                logger.warn(f"Unknown param argument {param}!")
            else:
                # check if param is within limits
                lower_limit, upper_limit = (
                    self.parameters[param].lower_limit,
                    self.parameters[param].upper_limit,
                )
                if param_args[param] < lower_limit:
                    param_args[param] = lower_limit
                    logger.warn(
                        f"Param argument {param} is below lower limit {lower_limit}! Setting it to {lower_limit}."
                    )
                elif param_args[param] > upper_limit:
                    param_args[param] = upper_limit
                    logger.warn(
                        f"Param argument {param} is above upper limit {upper_limit}! Setting it to {upper_limit}."
                    )

        return Action(self, obj_args, param_args, traj)


def create_action_from_raw(raw_action: str, primitive_actions: Dict[str, PrimitiveAction]):
    # [name, obj_args, param_1, param_2, ..]
    split = re.split("\(|\)|\{|\}|,", raw_action)
    split = [x.strip() for x in split if len(x) > 0 and len(x.strip()) > 0]  # remove space

    # get primitive action
    primitive_action = primitive_actions[split[0]]

    # get obj args
    obj_args = list(eval(split[1]))
    # get param args
    params = "{" + ", ".join(split[2:]) + "}"
    param_args = eval(params)

    return primitive_action(obj_args=obj_args, param_args=param_args)


def text_to_actions(text: List[str], primitive_actions: Dict[str, PrimitiveAction]):
    action_plan = []

    for raw_action in text:
        action = create_action_from_raw(raw_action, primitive_actions)
        action_plan.append(action)

    return action_plan
