import random
import numpy as np
from itertools import product

from pybullet_planning.utils import CIRCULAR_LIMITS, DEFAULT_RESOLUTION, MAX_DISTANCE
from pybullet_planning.interfaces.env_manager.pose_transformation import circular_difference, get_unit_vector, convex_combination

from pybullet_planning.interfaces.env_manager.user_io import wait_for_user
from pybullet_planning.interfaces.debug_utils import add_line
from pybullet_planning.interfaces.robots.joint import get_custom_limits, get_joint_positions,set_joint_positions
from pybullet_planning.interfaces.robots.collision import get_collision_fn

from pybullet_planning.motion_planners import birrt, lazy_prm

#####################################

# from utils.collision_util import get_collision_fn

# Joint motion planning

def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)

def halton_generator(d):
    import ghalton
    seed = random.randint(0, 1000)
    #sequencer = ghalton.Halton(d)
    sequencer = ghalton.GeneralizedHalton(d, seed)
    #sequencer.reset()
    while True:
        [weights] = sequencer.get(1)
        yield np.array(weights)

def unit_generator(d, use_halton=False):
    if use_halton:
        try:
            import ghalton
        except ImportError:
            print('ghalton is not installed (https://pypi.org/project/ghalton/)')
            use_halton = False
    return halton_generator(d) if use_halton else uniform_generator(d)

def interval_generator(lower, upper, **kwargs):
    assert len(lower) == len(upper)
    assert np.less_equal(lower, upper).all()
    if np.equal(lower, upper).all():
        return iter([lower])
    return (convex_combination(lower, upper, w=weights) for weights in unit_generator(d=len(lower), **kwargs))

def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
    generator = interval_generator(lower_limits, upper_limits, **kwargs)
    def fn():
        return tuple(next(generator))
    return fn

def get_halton_sample_fn(body, joints, **kwargs):
    return get_sample_fn(body, joints, use_halton=True, **kwargs)

def get_difference_fn(body, joints):
    from pybullet_planning.interfaces.robots.joint import is_circular
    circular_joints = [is_circular(body, joint) for joint in joints]

    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                     for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn

def get_distance_fn(body, joints, weights=None): #, norm=2):
    # TODO: use the energy resulting from the mass matrix here?
    if weights is None:
        weights = 1*np.ones(len(joints)) # TODO: use velocities here
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
        #return np.linalg.norm(np.multiply(weights * diff), ord=norm)
    return fn

def get_refine_fn(body, joints, num_steps=0):
    difference_fn = get_difference_fn(body, joints)
    num_steps = num_steps + 1
    def fn(q1, q2):
        q = q1
        yield q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            #q = tuple(wrap_positions(body, joints, positions))
            yield q
    return fn

def refine_path(body, joints, waypoints, num_steps):
    refine_fn = get_refine_fn(body, joints, num_steps)
    refined_path = []
    for v1, v2 in zip(waypoints, waypoints[1:]):
        refined_path += list(refine_fn(v1, v2))
    return refined_path

def get_extend_fn(body, joints, resolutions=None, norm=2):
    # norm = 1, 2, INF
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION*np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)
    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.ceil(np.linalg.norm(np.divide(difference_fn(q2, q1), resolutions), ord=norm)))
        refine_fn = get_refine_fn(body, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn

def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path

def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path
    difference_fn = lambda q2, q1: np.array(q2) - np.array(q1)
    #difference_fn = get_difference_fn(body, joints)

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints

def adjust_path(robot, joints, path):
    start_positions = get_joint_positions(robot, joints)
    difference_fn = get_difference_fn(robot, joints)
    differences = [difference_fn(q2, q1) for q1, q2 in zip(path, path[1:])]
    adjusted_path = [np.array(start_positions)]
    for difference in differences:
        adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path


def plan_waypoints_joint_motion(body, joints, waypoints, start_conf=None, obstacles=[], attachments=[],
                                self_collisions=True, disabled_collisions=set(),
                                resolutions=None, custom_limits={}, max_distance=MAX_DISTANCE):
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    else:
        assert len(start_conf) == len(joints)

    for i, waypoint in enumerate([start_conf] + list(waypoints)):
        if collision_fn(waypoint):
            #print("Warning: waypoint configuration {}/{} is in collision".format(i, len(waypoints)))
            return None
    path = [start_conf]
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path

def plan_direct_joint_motion(body, joints, end_conf, **kwargs):
    """plan a joint trajectory connecting the robot's current conf to the end_conf

    Parameters
    ----------
    body : [type]
        [description]
    joints : [type]
        [description]
    end_conf : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return plan_waypoints_joint_motion(body, joints, [end_conf], **kwargs)

def get_detailed_ee_collision_func(body, joints, ee_link, obstacles=[], attachments=[], body_name_from_id=None, **kwargs):
    """get collision checking function collision_fn(joint_values) -> bool.

    Parameters
    ----------
    body : int
        the robot body
    joints : list of int
        moving joint indices for body
    ee_link : int
        end effector link index
    obstacles : list of int
        body indices for collision objects, by default []
    attachments : list of Attachment, optional
        list of attachment, by default []

    Returns
    -------
    function handle
        collision_fn: (conf, diagnosis) -> False if no collision found, True otherwise.
        if need diagnosis information for the collision, set diagnosis to True will help you visualize
        which link is colliding to which.
    """
    from pybullet_planning.interfaces.robots.joint import set_joint_positions
    from pybullet_planning.interfaces.debug_utils.debug_utils import draw_collision_diagnosis
    from pybullet_planning.interfaces.robots.collision import expand_links, pairwise_link_collision, pairwise_link_collision_info
    from pybullet_planning.interfaces.robots.link import get_link_name
    from pybullet_planning.interfaces.robots.body import get_body_name
    moving_links = [ee_link]
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [(body, moving_links)] + attached_bodies
    # * body pairs
    check_body_pairs = list(product(moving_bodies, obstacles))  # + list(combinations(moving_bodies, 2))
    check_body_link_pairs = []
    for body1, body2 in check_body_pairs:
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        if body1 == body2:
            continue
        bb_link_pairs = product(links1, links2)
        for bb_links in bb_link_pairs:
            bbll_pair = ((body1, bb_links[0]), (body2, bb_links[1]))
            check_body_link_pairs.append(bbll_pair)

    def detailed_ee_collision_fn(q, diagnosis=False):
        # * set body & attachment positions
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()

        # * body - body check
        for (body1, link1), (body2, link2) in check_body_link_pairs:
            if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
                if diagnosis:
                    # warnings.warn('moving body - body collision!', UserWarning)
                    cr = pairwise_link_collision_info(body1, link1, body2, link2)
                    draw_collision_diagnosis(cr, body_name_from_id=body_name_from_id, **kwargs)
                return True, body2
        return False, None
    return detailed_ee_collision_fn

def check_ee_collision(body, joints, ee_link, end_conf, obstacles={}, attachments=[], diagnosis=False):
    start_conf = get_joint_positions(body, joints)
    detailed_ee_collision_fn = get_detailed_ee_collision_func(body, joints, ee_link, list(obstacles.keys()), attachments)
    feedback = ""

    if_collision, collide_obstacle = detailed_ee_collision_fn(end_conf, diagnosis)
    # set to start_conf
    set_joint_positions(body, joints, start_conf)

    if if_collision:
        feedback = f"Failed because the end configuration is in collision with object {obstacles[collide_obstacle]}"

    return if_collision, feedback

def plan_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(), extra_disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, diagnosis=False, **kwargs):
    """call birrt to plan a joint trajectory from the robot's **current** conf to ``end_conf``.
    """
    print("Planning joint motion...")
    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles=obstacles, attachments=attachments, self_collisions=self_collisions,
                                    disabled_collisions=disabled_collisions, extra_disabled_collisions=extra_disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)

    start_conf = get_joint_positions(body, joints)    
    birrt_res = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)
    feedback = "" if birrt_res is not None else "Failed because no collision-free trajectory is found"

    # set to start_conf
    set_joint_positions(body, joints, start_conf)
    
    return birrt_res, feedback
    # return plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn)                # better

def plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn, **kwargs):
    # TODO: cost metric based on total robot movement (encouraging greater distances possibly)
    path, samples, edges, colliding_vertices, colliding_edges = lazy_prm(
        start_conf, end_conf, sample_fn, extend_fn, collision_fn, num_samples=200, **kwargs)
    if path is None:
        return path

    #lower, upper = get_custom_limits(body, joints, circular_limits=CIRCULAR_LIMITS)
    def draw_fn(q): # TODO: draw edges instead of vertices
        return np.append(q[:2], [1e-3])
        #return np.array([1, 1, 0.25])*(q + np.array([0., 0., np.pi]))
    handles = []
    for q1, q2 in zip(path, path[1:]):
        handles.append(add_line(draw_fn(q1), draw_fn(q2), color=(0, 1, 0)))
    for i1, i2 in edges:
        color = (0, 0, 1)
        if any(colliding_vertices.get(i, False) for i in (i1, i2)) or colliding_vertices.get((i1, i2), False):
            color = (1, 0, 0)
        elif not colliding_vertices.get((i1, i2), True):
            color = (0, 0, 0)
        handles.append(add_line(draw_fn(samples[i1]), draw_fn(samples[i2]), color=color))
    # wait_for_user()
    return path
