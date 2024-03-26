import os
import abc
import logging
import numpy as np
from scipy.spatial.transform import Rotation

import pybullet as p
from pybullet_planning import (
    connect,
    disconnect,
    is_connected,
    create_obj,
    set_pose,
    Pose,
    load_pybullet,
    get_movable_joints,
    link_from_name,
    set_joint_positions,
    get_joint_positions,
    create_attachment,
    get_link_pose,
    get_distance,
    quat_angle_between,
    wait_for_duration,
    multiply,
)

from utils.planning_util import plan_joint_motion, check_ee_collision
from utils.pb_util import (
    assign_link_colors,
    create_box,
    set_point,
    get_pose,
    set_static,
    add_data_path,
)
from utils.tamp_util import Action, TAMPFeedback
from envs.constants import ASSETS_DIR, COLOR_FRANKA, FRANKA_Limits, BROWN, TAN

logger = logging.getLogger(__name__)


BOX_PATH = os.path.join(ASSETS_DIR, "box_obstacle3.obj")
HOOK_PATH = os.path.join(ASSETS_DIR, "hook.obj")
FRANKA_ROBOT_URDF = os.path.join(ASSETS_DIR, "franka_description", "robots", "panda_arm_hand.urdf")


class PybulletEnv:
    def __init__(self):
        self._objects = {}
        self._primitive_actions = {}
        self.theta_dict = {}

    @property
    def objects(self):
        return self._objects

    @property
    def primitive_actions(self):
        return self._primitive_actions

    @property
    def body_id_name_mapping(self):
        return {v: k for k, v in self._objects.items()}

    def reset(self, use_gui=True):
        # destroy the current simulation
        self.destroy()

        # connect to pybullet
        connect(use_gui=use_gui)
        p.setGravity(0, 0, -10)

        # add floor
        add_data_path()
        p.loadURDF("plane.urdf")

        # reset objects
        self._objects = {}

        # add pybullet robot
        self.robot = PyBulletRobot()

        # reset camera
        p.resetDebugVisualizerCamera(
            cameraDistance=2.6,
            cameraYaw=40,
            cameraPitch=-45,
            cameraTargetPosition=[-0.3, 0.5, -1],
        )

    def destroy(self):
        if is_connected():
            disconnect()

    def simulate(self, num_steps=100):
        for _ in range(num_steps):
            p.stepSimulation()

    def collision_function(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        if xmin1 < xmax2 and xmax1 > xmin2 and ymin1 < ymax2 and ymax1 > ymin2:
            return True
        return False

    def step(self, action: Action, *args, **kwargs):
        # apply action
        success, mp_feedback = self.apply_action(action, *args, **kwargs)

        # get observation
        observation = self.get_observation()

        # check goal
        goal_achieved, goal_feedback = self.check_goal()

        # prepare feedback
        feedback = TAMPFeedback(
            motion_planner_feedback=mp_feedback,
            task_process_feedback=goal_feedback,
            action_success=success,
            goal_achieved=goal_achieved,
        )
        logger.info(feedback.motion_planner_feedback)

        if goal_achieved:
            logger.debug("Goal achieved!")

        return observation, feedback

    def get_observation(self):
        # in this general function, we assume getting observations for every objects
        # cprint("here is in the observation function", "yellow")
        observation = {}
        for obj_name, obj in self.objects.items():
            # for position
            pos = self.get_position(obj_name)
            # bbox = self.get_bounding_box(obj_name)
            bb_min, bb_max = self.get_bb(obj_name)

            observation[obj_name] = {"position": pos, "bb_min": bb_min, "bb_max": bb_max}
        return observation

    @abc.abstractmethod
    def apply_action(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def check_goal(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def create_task_instances(self):
        raise NotImplementedError("Override me!")

    def create_table(self, name="table", color=TAN, x=0.1, y=0, z=0):  # -0.005):
        table_body = create_box(w=3, l=3, h=0.01, color=color)
        set_point(table_body, [x, y, z])
        self._objects[name] = table_body

    def create_basket(self, name="basket", color=BROWN, x=0.6, y=0, z=0.002, w=0.2, l=0.4, h=0.01):
        basket_body = create_box(w=w, l=l, h=h, color=color)
        set_point(basket_body, [x, y, z])
        self._objects[name] = basket_body

    def create_customized_box(self, name, color, w, l, h, x, y, z, theta=np.pi):
        body = create_box(w=w, l=l, h=h, color=color, mass=1.0)
        set_pose(body, Pose(point=[x, y, z], euler=np.array([theta, 0, 0])))
        self._objects[name] = body
        self.theta_dict[name] = 0

    def create_box(self, name, color, x, y, z, theta=np.pi):
        box_body = create_obj(BOX_PATH, scale=0.5, color=color, mass=1.0)
        set_pose(box_body, Pose(point=[x, y, z], euler=np.array([theta, 0, 0])))
        self._objects[name] = box_body
        self.theta_dict[name] = 0

    def create_tool(self, name, color, x, y, z, theta=np.pi):
        tool_body = create_obj(HOOK_PATH, scale=0.3, color=color, mass=1.0)
        set_pose(tool_body, Pose(point=[x, y, z], euler=np.array([theta, 0, 0])))
        self._objects[name] = tool_body

    def get_position(self, name):
        return get_pose(self._objects[name])[0]

    def get_orientation(self, name):
        return get_pose(self._objects[name])[1]

    def get_bb(self, name):
        return p.getAABB(self._objects[name])

    def prepare_obstacles(self, obj_name_list=[], remove_mode=False):
        obstacles = {}
        # if remove mode, remove objects in obj_name_list
        if remove_mode:
            for obj_name, id in self._objects.items():
                if obj_name not in obj_name_list:
                    obstacles[id] = obj_name
        else:
            # add obstacles
            for obj_name in obj_name_list:
                obstacles[self._objects[obj_name]] = obj_name

        return obstacles


class PyBulletRobot:
    def __init__(self):
        # load robot: Franka FR3 for now
        self.robot = load_pybullet(FRANKA_ROBOT_URDF, fixed_base=True)
        assign_link_colors(self.robot, colors=COLOR_FRANKA)
        self.ik_joints = get_movable_joints(self.robot)
        self.tool_attach_link = link_from_name(self.robot, "tool_link")

        # set static: important that the robot is static during planning
        set_static(self.robot)

        # grasping attachments & direction
        self.attachments_robot = []
        self.last_grasp_direction = None

        # set grasping offset: position and orientation
        self.position_offset_dict = {
            "top": np.array([0.0, 0, 0.05]),
            "left": np.array([0.5, -0.15, 0]),
            "right": np.array([0, 0.08, 0]),
            "forward": np.array([-0.1, 0, 0]),
        }

        x_axis_positive = [0, 0.7071068, 0, 0.7071068]
        x_axis_negative = [0, -0.7071068, 0, 0.7071068]
        y_axis_negative = [0.7068252, 0, 0, 0.7073883]
        y_axis_positive = [-0.7068252, 0, 0, 0.7073883]
        z_axis_positive = [0, 0, 0, 1]
        z_axis_negative = [1, 0, 0, 0]
        self.rotation_offset_dict = {
            "top": z_axis_negative,
            "left": y_axis_positive,
            "right": y_axis_negative,
            "forward": x_axis_positive,
            "backward": x_axis_negative,
        }

        # initialize pose
        self.initialize_pose()

    def initialize_pose(self):
        home_conf = [0, -0.785398163397, 0, -2.35619449, 0, 1.57079632679, 0.78539816, 0.04, 0.04]
        set_joint_positions(self.robot, self.ik_joints, home_conf)

    def pick(self, object, obstacles, grasp_direction, traj=None, play_traj=True):
        assert grasp_direction in self.position_offset_dict.keys(), "Unknown grasp direction!"
        if len(self.attachments_robot) > 0:
            self.release_gripper()
        # prepare grasping pose
        current_conf = get_joint_positions(self.robot, self.ik_joints)
        box_position, box_orientation = get_pose(object)
        ee_grasp_position = box_position + self.position_offset_dict[grasp_direction]
        ee_grasp_orientation = self.rotation_offset_dict[grasp_direction]

        if traj is None:
            success, traj, feedback = self.motion_planning(
                self.robot,
                self.ik_joints,
                ee_grasp_position,
                ee_grasp_orientation,
                obstacles,
                self.attachments_robot,
                FRANKA_Limits,
            )
        else:
            assert (
                traj[0] == current_conf
            ), f"The start conf of the known trajectory {traj[0]} is not the same as the robot start conf {current_conf}"

            success = True
            feedback = "Success"

        # update attachments
        if success:
            # simulate action
            ee_link_from_tcp = Pose(point=(0, 0.00, 0.0))
            self.simulate_traj(
                self.robot,
                self.ik_joints,
                self.attachments_robot,
                None,
                self.tool_attach_link,
                ee_link_from_tcp,
                traj,
                play_traj,
            )

            lifted_position = [box_position[0], box_position[1], box_position[2] + 0.01]
            set_point(object, lifted_position)
            box_attach = create_attachment(self.robot, self.tool_attach_link, object)
            box_attach.assign()
            self.attachments_robot.append(box_attach)
            logger.debug(f"box_position: {box_position}")

            # set last grasp direction
            self.last_grasp_direction = grasp_direction

        return success, traj, feedback

    def place(self, object, obstacles, x, y, z, theta, traj=None, play_traj=True):
        assert len(self.attachments_robot) > 0, "No object attached!"

        current_conf = get_joint_positions(self.robot, self.ik_joints)
        box_position, box_orientation = get_pose(object)
        logger.debug(f"box_position: {box_position}")
        logger.debug(f"box_orientation: {box_orientation}")

        new_box_position = (x, y, z)
        # new_box_orientation = box_orientation

        rot = Rotation.from_euler("XYZ", [np.pi, 0, theta], degrees=False)
        quat = rot.as_quat()

        position_offset = self.position_offset_dict[self.last_grasp_direction]
        ee_grasp_position = new_box_position + position_offset

        # rotation_offset = self.rotation_offset_dict[self.last_grasp_direction]
        ee_grasp_orientation = quat

        if traj is None:
            success, traj, feedback = self.motion_planning(
                self.robot,
                self.ik_joints,
                ee_grasp_position,
                ee_grasp_orientation,
                obstacles,
                self.attachments_robot,
                FRANKA_Limits,
            )
        else:
            assert (
                traj[0] == current_conf
            ), f"The start conf of the known trajectory {traj[0]} is not the same as the robot start conf {current_conf}"

            success = True
            feedback = "Success"

        # release gripper
        if success:
            ee_link_from_tcp = Pose(point=(0, 0.00, 0.05))
            # simulate action
            self.simulate_traj(
                self.robot,
                self.ik_joints,
                self.attachments_robot,
                object,
                self.tool_attach_link,
                ee_link_from_tcp,
                traj,
                play_traj,
            )

            self.release_gripper()
            logger.debug(f"Placed object {object}!")

        return success, traj, feedback

    def release_gripper(self):
        self.attachments_robot = []
        self.last_grasp_direction = None

    def verify_ik(self, targeted_pos, targeted_ori, joint_conf):
        original_conf = get_joint_positions(self.robot, self.ik_joints)
        set_joint_positions(self.robot, self.ik_joints, joint_conf)
        pos, ori = get_link_pose(self.robot, self.tool_attach_link)

        dist = get_distance(pos, targeted_pos)
        ori_dist = quat_angle_between(ori, targeted_ori)

        set_joint_positions(self.robot, self.ik_joints, original_conf)

        if dist < 0.01 and ori_dist < 0.4:
            # print("IK solution found")
            return True
        else:
            # print("IK solution not found")
            return False

    def motion_planning(
        self,
        robot,
        ik_joints,
        robot_ee_position,
        robot_end_orientation,
        obstacles,
        attachments_robot,
        custom_limits,
        diagnosis=False,
    ):
        # test ik first
        end_conf = p.calculateInverseKinematics(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self.tool_attach_link,
            targetPosition=robot_ee_position,
            targetOrientation=robot_end_orientation,
            # currentPosition=[0, -0.785398163397, 0, -2.35619449, 0, 1.57079632679, 0.78539816, 0.01, 0.01],
            maxNumIterations=100000,
            residualThreshold=0.0001,
        )

        ik_found = self.verify_ik(robot_ee_position, robot_end_orientation, end_conf)
        if not ik_found:
            logger.debug("Do not have a feasible ik")
            failure_feedback = "Failed because no IK solution exists for the grasping pose"
            return False, None, failure_feedback

        # test final ee pose if in collision
        if_collision, feedback = check_ee_collision(
            self.robot,
            self.ik_joints,
            self.tool_attach_link,
            end_conf,
            obstacles,
            self.attachments_robot,
        )
        if if_collision:
            logger.debug(feedback)
            return False, None, feedback

        # plan motion
        logger.debug(str(obstacles))
        path, feedback = plan_joint_motion(
            robot,
            ik_joints,
            end_conf,
            obstacles=list(obstacles.keys()),
            attachments=attachments_robot,
            self_collisions=True,
            custom_limits=custom_limits,
            diagnosis=diagnosis,
        )

        # process planned results
        if path is None:
            logger.debug(feedback)
            return False, None, feedback
        else:
            logger.debug("Found feasible motion plan!")

        return True, path, "Success"

    def simulate_traj(
        self,
        robot,
        ik_joints,
        attachments_robot,
        attached,
        tool_attach_link,
        ee_link_from_tcp,
        traj,
        play_traj,
    ):
        if play_traj:
            logger.debug("Simulate trajectory!")
            time_step = 0.03
            logger.debug(f"attachments_robot: {attachments_robot}")
            logger.debug(f"attached: {attached}")
            for conf in traj:
                set_joint_positions(robot, ik_joints, conf)
                if len(attachments_robot) > 0:
                    ee_link_pose = get_link_pose(robot, tool_attach_link)
                    set_pose(attached, multiply(ee_link_pose, ee_link_from_tcp))
                wait_for_duration(time_step)
        else:
            # directly set end conf
            set_joint_positions(robot, ik_joints, traj[-1])
            if len(attachments_robot) > 0:
                ee_link_pose = get_link_pose(robot, tool_attach_link)
                set_pose(attached, multiply(ee_link_pose, ee_link_from_tcp))
