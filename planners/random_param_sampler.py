import random
from typing import Dict, Tuple
from utils.tamp_util import PrimitiveAction


class RandomParamSampler:
    def __init__(
        self,
        primitive_actions: Dict[str, PrimitiveAction],
    ):
        self.primitive_actions = primitive_actions

    def reset(self, *args, **kwargs):
        pass

    def plan(self, x_range, y_range, theta_range, *args, **kwargs):
        # hard code at this moment
        sampled_x, sampled_y, sampled_theta = [], [], []
        for _ in range(4):
            sampled_x.append(random.uniform(x_range[0], x_range[1]))
            sampled_y.append(random.uniform(y_range[0], y_range[1]))
            sampled_theta.append(random.uniform(theta_range[0], theta_range[1]))

        return [
            self.primitive_actions["pick"](["red_box"], {}),
            self.primitive_actions["place"](
                ["red_box"], {"x": sampled_x[0], "y": sampled_y[0], "theta": sampled_theta[0]}
            ),
            self.primitive_actions["pick"](["blue_box"], {}),
            self.primitive_actions["place"](
                ["blue_box"], {"x": sampled_x[1], "y": sampled_y[1], "theta": sampled_theta[1]}
            ),
            self.primitive_actions["pick"](["green_box"], {}),
            self.primitive_actions["place"](
                ["green_box"], {"x": sampled_x[2], "y": sampled_y[2], "theta": sampled_theta[2]}
            ),
            self.primitive_actions["pick"](["tan_box"], {}),
            self.primitive_actions["place"](
                ["tan_box"], {"x": sampled_x[3], "y": sampled_y[3], "theta": sampled_theta[3]}
            ),
        ]
