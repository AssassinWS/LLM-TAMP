import logging
import numpy as np
from pathlib import Path
import pybullet as p
from envs.pack_compact_env import PackCompactEnv
from planners.llm_tamp_planner import LLMTAMPPlanner
from planners.random_param_sampler import RandomParamSampler

from utils.io_util import mkdir, save_npz, dump_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TAMPRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cfg = cfg.env
        self.planner_cfg = cfg.planner

        # environment
        self.env = PackCompactEnv()
        self.primitive_actions = self.env.primitive_actions

        # save dirs
        self.save_to_file = cfg.save_to_file
        self.world_dir = Path("envs/task_instances")
        mkdir(self.world_dir)
        self.save_dir = Path(cfg.save_dir)

        # planner
        self.planner = LLMTAMPPlanner(
            planner_prompt_file=self.planner_cfg.planner_prompt_file,
            env_desc_file="pack_boxes.txt",
            primitive_actions=self.primitive_actions,
            with_mp_feedback=self.planner_cfg.with_mp_feedback,
            trace_size=self.planner_cfg.trace_size,
        )
        self.max_llm_calls = cfg.max_llm_calls

        self.play_traj = cfg.play_traj
        self.use_gui = cfg.use_gui

        logger.info(f"Run TAMP for setting {cfg.env.env_name}!")

    def run_once(self, task_config):
        # main loop
        last_feedback_list = []
        last_temp_tamp_plan = None
        final_tamp_plan = None
        num_mp_calls = 0
        num_llm_calls = 0
        for _ in range(self.max_llm_calls):
            # reset environment
            obs, obs_text = self.env.reset(**task_config, use_gui=self.use_gui)
            # propose plan with llm (symbolic plan only used when sampling parameters only)
            plan = self.planner.plan(
                obs_text, last_feedback_list, symbolic_plan=self.env.get_symbolic_plan()
            )
            last_feedback_list = []  # last feedback
            num_llm_calls += 1

            # rollout
            temp_tamp_plan = []
            same_as_last = True
            for action_i, action in enumerate(plan):
                # if same as last, simulate last traj
                if (
                    same_as_last
                    and last_temp_tamp_plan is not None
                    and len(last_temp_tamp_plan) > action_i
                ):
                    if str(action) == str(last_temp_tamp_plan[action_i]):
                        action = last_temp_tamp_plan[action_i]
                    else:
                        same_as_last = False

                # motion planning when no traj
                if action.traj is None or len(action.traj) == 0:
                    num_mp_calls += 1

                _, feedback = self.env.step(
                    action, play_traj=self.play_traj
                )  # this step will also save traj in action
                last_feedback_list.append((action, feedback))

                logger.debug(f"Apply action: {action}")
                logger.debug(f"Succeed: {feedback.action_success}")
                logger.debug(f"MP feedback: {feedback.motion_planner_feedback}")

                if feedback.action_success:
                    temp_tamp_plan.append(action)
                else:
                    logger.info(f"Action {str(action)} failed!")
                    break

                if feedback.goal_achieved:
                    final_tamp_plan = temp_tamp_plan
                    break

            last_temp_tamp_plan = temp_tamp_plan
            if feedback.goal_achieved:
                logger.info("Find full plan!")
                break
            else:
                logger.info(f"Goal not achieved: {feedback.task_process_feedback}")

        logger.info("Episode ends!")
        self.env.destroy()

        episode_data = {
            "tamp_plan": final_tamp_plan,
            "goal_achieved": feedback.goal_achieved,
            "num_mp_calls": num_mp_calls,
            "num_llm_calls": num_llm_calls,
        }

        return episode_data

    def run(self):
        task_instances = self.env.create_task_instances(
            self.env_cfg,
            self.env_cfg.task_instances,
            save_to_file=self.save_to_file,
            instance_file=self.world_dir / f"{self.env_cfg.env_name}.json",
            overwrite=self.cfg.overwrite_instances,
        )

        goal_achieved_list = []
        num_steps_list = []
        num_mp_calls_list = []
        num_llm_calls_list = []
        for idx, task_config in task_instances.items():
            # reset planner
            self.planner.reset()

            episode_data = self.run_once(task_config)

            goal_achieved = episode_data["goal_achieved"]
            num_llm_calls = episode_data["num_llm_calls"]
            num_mp_calls = episode_data["num_mp_calls"]

            logger.info(f"Goal achieved: {goal_achieved}")
            logger.info(f"Number of MP calls: {num_mp_calls}")
            logger.info(f"Number of LLM calls: {num_llm_calls}")

            goal_achieved_list.append(goal_achieved)

            if goal_achieved:
                num_steps_list.append(len(episode_data["tamp_plan"]))
            else:
                num_steps_list.append(-1)
            num_mp_calls_list.append(episode_data["num_mp_calls"])
            num_llm_calls_list.append(episode_data["num_llm_calls"])

            if self.save_to_file:
                # save tamp_plan into npz
                save_episode_dir = self.save_dir / f"{idx}"
                mkdir(save_episode_dir)
                # import pdb

                # pdb.set_trace()
                save_npz(episode_data, save_episode_dir / "result.npz")

                # save json every time
                json_data = {
                    "success_rate": np.mean(goal_achieved_list),
                    "goal_achieved": goal_achieved_list,
                    "num_steps": num_steps_list,
                    "num_mp_calls": num_mp_calls_list,
                    "num_llm_calls": num_llm_calls_list,
                }
                dump_json(json_data, self.save_dir / "result.json")


class RandomSampleRunner(TAMPRunner):
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cfg = cfg.env

        # environment
        self.env = PackCompactEnv()
        self.primitive_actions = self.env.primitive_actions

        # save dirs
        self.save_to_file = cfg.save_to_file
        self.world_dir = Path("envs/task_instances")
        mkdir(self.world_dir)
        self.save_dir = Path(cfg.save_dir)

        # planner
        self.planner = RandomParamSampler(primitive_actions=self.primitive_actions)
        self.max_sample_iters = cfg.max_sample_iters

        self.play_traj = cfg.play_traj
        self.use_gui = cfg.use_gui

        logger.info(f"Run parameter sampling for setting {cfg.env.env_name}!")

    def run_once(self, task_config):
        # main loop
        last_temp_tamp_plan = None
        final_tamp_plan = None
        num_mp_calls = 0
        num_sample_iters = 0
        while True:
            # reset environment
            obs, obs_text = self.env.reset(**task_config, use_gui=self.use_gui)
            bb_min, bb_max = self.env.get_bb("basket")
            x_range = [bb_min[0], bb_max[0]]
            y_range = [bb_min[1], bb_max[1]]

            # x_range = [0, 1]
            # y_range = [-1, 1]
            theta_range = [-np.pi, np.pi]

            # propose plan with llm (symbolic plan only used when sampling parameters only)
            plan = self.planner.plan(x_range, y_range, theta_range)
            num_sample_iters += 1

            # rollout
            temp_tamp_plan = []
            same_as_last = True
            for action_i, action in enumerate(plan):
                # if same as last, simulate last traj
                if (
                    same_as_last
                    and last_temp_tamp_plan is not None
                    and len(last_temp_tamp_plan) > action_i
                ):
                    if str(action) == str(last_temp_tamp_plan[action_i]):
                        action = last_temp_tamp_plan[action_i]
                    else:
                        same_as_last = False

                # motion planning when no traj
                if action.traj is None or len(action.traj) == 0:
                    num_mp_calls += 1

                _, feedback = self.env.step(
                    action, play_traj=self.play_traj
                )  # this step will also save traj in action

                logger.debug(f"Apply action: {action}")
                logger.debug(f"Succeed: {feedback.action_success}")

                if feedback.action_success:
                    temp_tamp_plan.append(action)
                else:
                    logger.info(f"Action {str(action)} failed!")
                    break

                if feedback.goal_achieved:
                    final_tamp_plan = temp_tamp_plan
                    break

            last_temp_tamp_plan = temp_tamp_plan
            if feedback.goal_achieved:
                logger.info("Find full plan!")
                break
            else:
                logger.info(f"Goal not achieved: {feedback.task_process_feedback}")

            if num_sample_iters >= self.max_sample_iters:
                logger.info("Reach max sample iters!")
                break

        logger.info("Episode ends!")
        self.env.destroy()

        episode_data = {
            "tamp_plan": final_tamp_plan,
            "goal_achieved": feedback.goal_achieved,
            "num_mp_calls": num_mp_calls,
            "num_sample_iters": num_sample_iters,
        }

        return episode_data

    def run(self):
        task_instances = self.env.create_task_instances(
            self.env_cfg,
            self.env_cfg.task_instances,
            save_to_file=self.save_to_file,
            instance_file=self.world_dir / f"{self.env_cfg.env_name}.json",
            overwrite=self.cfg.overwrite_instances,
        )

        goal_achieved_list = []
        num_steps_list = []
        num_mp_calls_list = []
        num_sample_iters_list = []
        for idx, task_config in task_instances.items():
            episode_data = self.run_once(task_config)

            goal_achieved = episode_data["goal_achieved"]
            num_sample_iters = episode_data["num_sample_iters"]
            num_mp_calls = episode_data["num_mp_calls"]

            logger.info(f"Goal achieved: {goal_achieved}")
            logger.info(f"Number of MP calls: {num_mp_calls}")
            logger.info(f"Number of sample iters: {num_sample_iters}")

            goal_achieved_list.append(goal_achieved)

            if goal_achieved:
                num_steps_list.append(len(episode_data["tamp_plan"]))
            else:
                num_steps_list.append(-1)
            num_mp_calls_list.append(episode_data["num_mp_calls"])
            num_sample_iters_list.append(episode_data["num_sample_iters"])

            if self.save_to_file:
                # save tamp_plan into npz
                save_episode_dir = self.save_dir / f"{idx}"
                mkdir(save_episode_dir)
                save_npz(episode_data, save_episode_dir / "result.npz")

        if self.save_to_file:
            json_data = {
                "success_rate": np.mean(goal_achieved_list),
                "goal_achieved": goal_achieved_list,
                "num_steps": num_steps_list,
                "num_mp_calls": num_mp_calls_list,
                "num_sample_iters": num_sample_iters_list,
            }
            dump_json(json_data, self.save_dir / "result.json")
