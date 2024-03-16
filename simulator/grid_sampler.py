import os
import cv2
import json
import itertools
import quaternion # Remove this will cause invalid pointer error !!!!
import habitat_sim
import numpy as np
from tqdm import tqdm
from utils import config
from utils.dataset_interface import Objaverse, HM3D
from simulator.multisensory_simulator import MultisensorySimulator
from utils.config import sim_conf
from utils.cloud_point_utils import Reconstruct3D, crop_points


class GridSampler(MultisensorySimulator):
    def __init__(self, scene, new_objs=None, audio=False, rooms=None):
        self.scene = scene
        self.grid_points = np.load(os.path.join(config.SAMPLE_DIR, self.scene, "grid_points.npy"))
        self.room_boxes = []
        if rooms is not None:
            hm3d = HM3D()
            for i in rooms:
                bboxes = hm3d.load_room(f"{scene}_{i}.json")
                _min, _max = hm3d.room_bbox(bboxes)
                _min = [_min[0], _min[2], _min[1]]
                _max = [_max[0], _max[2], _max[1]]
                self.room_boxes.append([_min, _max])
            self.grid_points = crop_points(self.grid_points, self.room_boxes)
            print(self.room_boxes)

        action_space = {
            "look_left": habitat_sim.agent.ActionSpec("look_left", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_up": habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_down": habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=90.0))}
        cfg = sim_conf(scene, audio=audio)
        cfg.agents[0].action_space.update(action_space)
        super().__init__(cfg, new_objs)

        _spec = cfg.agents[0].sensor_specifications[0]
        self.reconstructor = Reconstruct3D(
            _spec.resolution[0],
            _spec.resolution[1],
            float(_spec.hfov),
            _spec.position
        )

    # Return:
    # agent_location -> grid_points.npy
    # camera_direction -> [left0, left90, left180, left270, up90, down90]
    def scan_scene(self):
        quats = [self.degree2quat(*x) for x in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]]

        points = []
        for i in tqdm([x for x in self.grid_points if self.pathfinder.is_navigable(x)]):
            new_state = habitat_sim.AgentState(i, [0, 0, 0, 1])
            self.agents[0].set_state(new_state) # set position

            # Scan in sphere
            obs = [self.parse_visual_observation(self.get_sensor_observations()),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left"))]

            self.agents[0].set_state(new_state)  # reset
            obs.append(self.parse_visual_observation(self.step("look_up")))
            self.agents[0].set_state(new_state)  # reset
            obs.append(self.parse_visual_observation(self.step("look_down")))

            # convert to points
            coordinates = []
            valid_masks = []
            for o, q in zip(obs, quats):
                p, valid_mask = self.reconstructor.depth_map2points(o[:, :, 3], q, i)
                coordinates.append(p)
                valid_masks.append(valid_mask)
            coordinates = np.concatenate(coordinates, axis=0)
            valid_idx = np.where(np.concatenate(valid_masks, axis=0))[0]

            # Concat all
            obs = np.stack(obs, axis=0).reshape(-1, 5)
            _points = np.concatenate([coordinates, obs[:, :3], obs[:, 4:]], axis=1).astype(np.float16) # xzy, rgb, semantic
            _points = _points[valid_idx]
            _points = _points[np.random.choice(len(_points), int(len(_points) / 10), replace=False), :]
            points.append(_points)

        points = np.concatenate(points, axis=0)
        if len(self.room_boxes):
            points = crop_points(points, self.room_boxes)
        idx = self.reconstructor.downsample_index(points[:, :3])
        return points[idx, :]

    @staticmethod
    # Channels -> [r, g, b, depth, semantic]
    def parse_visual_observation(obs):
        rgb = cv2.cvtColor(obs["rgba"], cv2.COLOR_RGBA2RGB)
        frame = np.concatenate([rgb, obs["depth"][:, :, np.newaxis], obs["semantic"][:, :, np.newaxis]], axis=-1)
        return frame

    def get_semantic_labels(self):
        id2cate = dict()
        if self.new_objs is not None:
            for i in self.new_objs:
                id2cate[i["semantic_id"]] = i["cate"]
        with open(os.path.join(config.HM3D_DIR, self.scene, f"{self.scene.split('-')[1]}.semantic.txt"), "r") as f:
            a = f.readlines()
        for i in a[1:]:
            i = i.strip()
            if len(i):
                _id = int(i.split(",")[0])
                _cate = i.split(",")[2].strip('"')
                id2cate[_id] = _cate
        return id2cate

    @staticmethod
    def degree2quat(z=0, x=0):
        assert (z * x) == 0
        if z:
            half_radians = np.deg2rad(z) / 2.0
            around_z_axis = [0, np.sin(half_radians), 0, np.cos(half_radians)]  # anticlockwise
            return around_z_axis
        elif x:
            half_radians = np.deg2rad(x) / 2.0
            around_x_axis = [np.sin(half_radians), 0, 0, np.cos(half_radians)]  # up is positive direction
            return around_x_axis
        else:
            return [0, 0, 0, 1]

# def sample_rirs(table, fps):
#     hm3d = HM3D()
#     top_center = lambda x: [(x[0][0] + x[1][0])/2, x[1][2], (x[0][1] + x[1][1])/2]
#     for _scene, v in table.items():
#         new_objs = list(itertools.chain.from_iterable(v.values()))
#         existing_objs = hm3d.load_scene(_scene)
#         obj2loc = {x["obj"]: top_center(x["bbox"]) for x in new_objs}
#         obj2loc.update({x["id"]: top_center(x["bbox"]) for x in existing_objs})
#
#         cfg = sim_conf(_scene, visual=False, audio=True)
#         sim = MultisensorySimulator(cfg, fps=fps, new_objs=None)
#         rirs = dict()
#         for k, v in obj2loc.items():
#             sim.set_audio_source(v)
#             obs = grid_sampling(sim, _scene)
#
#             _rirs = []
#             for i in obs:
#                 _r = []
#                 for j in i:
#                     _r.append(j["audio_sensor"])
#                 _rirs.append(_r)
#             rirs[k] = _rirs
#         json.dump(rirs, open(os.path.join(config.SAMPLE_DIR, _scene, "rirs.json"), "w"))


if __name__ == "__main__":
    # TODO: audio sampler after task template & grid_point navigable - coord dict to rirs
    # TODO: - change loop in calculate_audio() and reverb from previous time step
    objaverse = Objaverse()
    scene = json.load(open(os.path.join(config.DATA_DIR, "scene.json"), "r"))
    objaverse.get_objects([x["obj"] for x in scene]) # Download objects

    table = dict()
    for i in scene:
        _scene = i["room"].split("_")[0]
        if _scene not in table: table[_scene] = dict()
        _trail = i["trail"]
        if _trail not in table[_scene]: table[_scene][_trail] = []
        i["path"] = objaverse.get_objects([i["obj"]])[i["obj"]]
        table[_scene][_trail].append(i)

    for _scene, v in table.items():
        for _trail, _objs in v.items():
            print(_scene, _trail, _objs)

            sampler = GridSampler(_scene, _objs, audio=False)
            # Semantic Labels
            _path = os.path.join(config.SAMPLE_DIR, _scene, f"{_trail}.json")
            json.dump(sampler.get_semantic_labels(), open(_path, "w"))

            # Sampling
            points = sampler.scan_scene()
            np.save(os.path.join(config.SAMPLE_DIR, _scene, f"{_trail}.npy"), points)
            # Visualize (reverse y)
            # with open(os.path.join(config.SAMPLE_DIR, _scene, f"{_trail}.txt"), "w") as file:
            #     file.write(f"{len(points)}\n")
            #     for p in points:
            #         file.write(f"{p[0]} {-p[2]} {p[1]} {p[3]} {p[4]} {p[5]}\n")

    # Not support material yet: https://github.com/facebookresearch/sound-spaces/issues/111
    # sim.set_material_file("audio_sensor", "data/HM3D/mp3d_material_config.json")
