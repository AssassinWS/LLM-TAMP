import numpy as np
import itertools
import abc

from typing import Dict, Any
from utils.tamp_util import PrimitiveAction


class BasePlanner:
    def __init__(self, primitive_actions: Dict[str, PrimitiveAction], *args, **kwargs):
        self._primitive_actions = primitive_actions

    def __str__(self):
        return self.__class__.__name__

    def sample_random_action(self, observation: Dict[str, Any], *args, **kwargs):
        objects = list(observation.keys())
        # remove basket
        objects.remove("basket")

        # sample primitive action
        primitive = np.random.choice(list(self._primitive_actions.values()))

        # sample object arguments: use itertools
        all_permutations = list(itertools.permutations(objects, arity=primitive.obj_arity))
        obj_args = list(np.random.choice(all_permutations))
        print(obj_args)

        # sample parameter arguments
        param_args = {}
        for param_name, param in primitive.parameters.items():
            param_args[param_name] = np.random.uniform(
                low=param.lower_limit, high=param.upper_limit
            )

        return primitive(obj_args=obj_args, param_args=param_args)

    @abc.abstractmethod
    def plan(self, observation: Dict[str, Any], *args, **kwargs):
        raise NotImplementedError("Override me!")
