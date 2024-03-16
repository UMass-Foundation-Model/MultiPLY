import math
import numpy as np
import magnum as mn
import habitat_sim
from simulator.multisensory_simulator import MultisensorySimulator
from utils.config import sim_conf
from utils.dataset_interface import Objaverse
from habitat_sim.utils import viz_utils as vut
from habitat_sim.utils.common import quat_rotate_vector, quat_to_magnum


class GrapeObject(MultisensorySimulator):
    def __init__(self, scene, new_objs=None):
        cfg = sim_conf(scene, audio=False, physics=True)
        super().__init__(cfg, new_objs)
        self.fetchable_objs = [x["obj_id"] for x in self.new_objs if "mass" in x]
        self.reachable_range = 1.
        self.reachable_degree = math.radians(180)
        self.rigid_obj_mgr = self.get_rigid_object_manager()
        self.fetched_obj = None
        self.obj2agent = 0.3
        super().step_physics(dt=2) # Let obj fail

    def fetch_object(self, obj_id: int) -> bool:
        if self.fetched_obj: return False
        if not self.check_in_range(obj_id): return False
        self.fetched_obj = self.rigid_obj_mgr.get_object_by_id(obj_id)
        self.fetched_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self._update_fetched_obj()
        return True

    def _update_fetched_obj(self):
        self.fetched_obj.translation = self.agent_center + quat_rotate_vector(self.agent_rot, [0, 0, -1]) * self.obj2agent
        self.fetched_obj.rotation = quat_to_magnum(self.agent_rot)

    def check_in_range(self, obj_id: int) -> bool:
        # Check obj exist and fetchable
        if obj_id not in self.fetchable_objs: return False
        obj = self.rigid_obj_mgr.get_object_by_id(obj_id)
        if obj is None: return False

        # Check obj in dist range
        obj_loc = obj.translation
        dist = np.linalg.norm(obj_loc - self.agent_center)
        print("Distance", dist, obj_loc)
        if dist > self.reachable_range: return False

        # Check obj in angle range
        normal_vector = quat_rotate_vector(self.agent_rot, [0, 0, -1])
        relative_vector = obj_loc - self.agent_center
        dot = normal_vector[0] * relative_vector[0] + normal_vector[2] * relative_vector[2]
        det = normal_vector[0] * relative_vector[2] - normal_vector[2] * relative_vector[0]
        angle = math.atan2(det, dot)
        print("Angle", angle, normal_vector, relative_vector)
        if abs(angle) > (self.reachable_degree / 2): return False
        return True

    @property
    def agent_center(self):
        return self.agent_loc + [0, self.agents[0].agent_config.height - 0.3, 0]

    def drop_object(self, drop_dist=0.) -> bool:
        if not self.fetched_obj: return False
        self.fetched_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        self.fetched_obj.translation += quat_rotate_vector(self.agent_rot, [0, 0, -1]) * drop_dist
        self.fetched_obj = None
        return True

    # Override step to update fetched object states
    def step(self, action, dt=0.016666666666666666):
        super().step(action, dt)
        if self.fetched_obj:
            self._update_fetched_obj()
        return self.get_sensor_observations()

    def step_physics(self, dt: float, scene_id: int = 0) -> None:
        super().step_physics(dt, scene_id)
        self.observations.append(self.get_sensor_observations())


if __name__ == "__main__":
    objaverse = Objaverse()
    _objs = [
        {"cate": "donut", "bbox": [[-6.0, -1.2, 1.0], [-6.0, -1.2, 1.0 + 0.075]],
         "obj": "dcb0d1c9b8be49e0945535fdd81c7525", "mass": 0.05},
    ]
    for i in _objs:
        i["path"] = objaverse.get_objects([i["obj"]])[i["obj"]]
    sim = GrapeObject('00800-TEEsavR23oF', _objs)

    sim.move_agent_to_target([-6.73648, 0.163378, -1.21183])
    for i in range(3):
        sim.step_physics(1. / 10)
    success = sim.fetch_object(sim.fetchable_objs[0])
    print(success)

    sim.move_agent_to_target([-0.797001, 0.163378, -2.39349])
    sim.drop_object(drop_dist=0.4)
    for i in range(20):
        sim.step_physics(1. / 10)
    vut.make_video(sim.observations, "rgba", "color", "../color.mp4", fps=10, open_vid=False)
