from envs.pb_env import PybulletEnv
from envs.constants import RED, BLUE, GREEN, BROWN, TAN, WHITE, GREY, YELLOW
from termcolor import cprint
from utils.tamp_util import Action, PrimitiveAction


class UseToolEnv(PybulletEnv):
    """
    An environment for packing boxes in a compact basket.
    Use top grasp.
    """

    def __init__(self):
        super().__init__()

        self._primitive_actions = {
            "pick": PrimitiveAction("pick", obj_arity=1),
            "place": PrimitiveAction("place", obj_arity=1, param_names=["x", "y", "theta"]),
        }

    def reset(self):
        # parent class
        super().reset()

        # create table
        self.create_table()
        # create basket
        # self.create_basket(x = 0.8, y = 0, w = 0.3, l = 0.2)

        # create boxes
        self.create_box(name="red_box", color=RED, x=0.6, y=-0.8, z=0.11)
        # self.create_box(name="blue_box", color=BLUE, x=0.4, y=0.3, z=0.11)
        # self.create_box(name="green_box", color=GREEN, x=0.4, y=-0.2, z=0.11)
        # self.create_box(name="tan_box", color=TAN, x=0.2, y=-0.4, z=0.11)
        # self.create_box(name="grey_box", color=GREY, x=0.2, y=-0.6, z=0.11)
        # self.create_box(name="yellow_box", color=YELLOW, x=0.2, y=-0.2, z=0.11)

        self.create_tool(name="screwdriver", color=BLUE, x=0.4, y=0.3, z=0.11)

        self.simulate()

    def step(self, action: Action):
        # todo: this function should return the following: 1) observation (object poses, robot state, etc), 2) whether goal is achieved
        # sanity check
        assert action.primitive in self._primitive_actions.values(), "Unknown primitive action!"
        for obj_name in action.obj_args:
            assert obj_name in self.objects, "Unknown object name!"

        if action.primitive.name == "pick":
            obj_name = action.obj_args[0]
            object = self.objects[obj_name]

            # prepare obstacles (avoid all other objects)
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)

            success = self.robot.pick(object, obstacles, grasp_direction="top")
            # don't simulate at pick
            if success[0]:
                cprint("Picked!", "green")
            else:
                cprint("pick is not executed!:{}".format(success[1]), "red")

        elif action.primitive.name == "place":
            obj_name = action.obj_args[0]
            object = self.objects[obj_name]

            # prepare obstacles (avoid all other objects)
            obstacles = self.prepare_obstacles(obj_name_list=[obj_name], remove_mode=True)

            success = self.robot.place(
                object,
                obstacles,
                x=action.param_args["x"],
                y=action.param_args["y"],
                z=0.1,
                theta=action.param_args["theta"],
            )
            # only simulate at successful place
            if success[0]:

                self.simulate()
                self.theta_dict[obj_name] = action.param_args["theta"]
                cprint("Simulated!", "green")
            else:
                cprint("place is not executed!:{}".format(success[1]), "red")

    def get_env_info(self):
        # todo: return env info, including the task, table information, basket information, object information, etc.
        cprint("Goal: all boxes are packed into the basket", "red")
        cprint("Table size: 2 x 2 x 0.1", "red")
        cprint("Basket size: 0.2 x 0.4 x 0.01", "red")
        cprint("Box size: 0.1 x 0.1 x 0.1", "red")
        # for obj_name, obj in self.objects.items():
        #     cprint("{} position: {}".format(obj_name, obj.get_position()), "red") 
        
        for obj_name, obj in self.objects.items():
            cprint("{} position: {}".format(obj_name, self.get_position(obj_name)), "red") 
        for obj_name, obj in self.objects.items():
            if obj_name != "basket" and obj_name != "table":
                cprint("{} theta: {}".format(obj_name, self.theta_dict[obj_name]), "red") 

        basket_x, basket_y, basket_z = self.get_position("basket")
        basket_bb_lx = basket_x - 0.5 * 0.2
        basket_bb_ux = basket_x + 0.5 * 0.2
        basket_bb_ly = basket_y - 0.5 * 0.4
        basket_bb_uy = basket_y + 0.5 * 0.4 
        cprint("Basket bounding box: ({}, {}), ({}, {})".format(basket_bb_lx, basket_bb_ly, basket_bb_ux, basket_bb_uy), "red")

    def is_goal_achieved(self):
        for obj_name, obj in self.objects.items():
            if obj_name != "basket" and obj_name != "table":
                basket_x, basket_y, basket_z = self.get_position("basket")
                basket_bb_lx = basket_x - 0.5 * 0.2
                basket_bb_ux = basket_x + 0.5 * 0.2
                basket_bb_ly = basket_y - 0.5 * 0.4
                basket_bb_uy = basket_y + 0.5 * 0.4 
                obj_x, obj_y, obj_z = self.get_position(obj_name)

                if obj_x < basket_bb_lx or obj_x > basket_bb_ux or obj_y < basket_bb_ly or obj_y > basket_bb_uy:
                    return False
        return True
