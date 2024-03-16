import os
import cv2
import json
import itertools
import quaternion # Remove this will cause invalid pointer error !!!!
import habitat_sim
import numpy as np
from tqdm import tqdm
from utils import config
from utils.dataset_interface import Objaverse, HM3D, ObjectFolder
from multisensory_simulator import MultisensorySimulator
from utils.config import sim_conf
from utils.cloud_point_utils import Reconstruct3D
from model.feature_encoder import LlaVa_Encoder
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from PIL import Image
import copy
from collections import defaultdict
import random

class GridSampler(MultisensorySimulator):
    def __init__(self, scene, new_objs=None, audio=False, encoder=None):
        action_space = {
            "look_left": habitat_sim.agent.ActionSpec("look_left", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_up": habitat_sim.agent.ActionSpec("look_up", habitat_sim.agent.ActuationSpec(amount=90.0)),
            "look_down": habitat_sim.agent.ActionSpec("look_down", habitat_sim.agent.ActuationSpec(amount=90.0))}
        cfg = sim_conf(scene, audio=audio)
        cfg.agents[0].action_space.update(action_space)

        super().__init__(cfg, new_objs)

        self.scene = scene
        _spec = cfg.agents[0].sensor_specifications[0]
        self.reconstructor = Reconstruct3D(
            _spec.resolution[0],
            _spec.resolution[1],
            float(_spec.hfov),
            _spec.position
        )
        self.encoder = encoder

    @staticmethod
    def inside_p(pt, box):
        if pt[0] < box[0][0] or pt[0] > box[1][0]: return False
        if pt[1] < box[0][1] or pt[1] > box[1][1]: return False
        if pt[2] < box[0][2] or pt[2] > box[1][2]: return False
        if box[1][0] - pt[0] < 0 or pt[0] - box[0][0] < 0: return False
        if box[1][2] - pt[2] < 0 or pt[2] - box[0][2] < 0: return False
        return True

    def scan_scene(self, bbox_file, return_features = True):
        scene = bbox_file.split("_")[0]+".json"
        room = bbox_file.replace(".json", "").split("_")[1]
        room_bbox = json.load(open(os.path.join(config.ROOM_BBOX_DIR, scene)))[room]
        room_bbox = [[room_bbox[0][0], room_bbox[0][2], room_bbox[0][1]], [room_bbox[1][0], room_bbox[1][2], room_bbox[1][1]]]
        grid_points = np.load(os.path.join(config.SAMPLE_DIR, self.scene, "grid_points.npy"))
        quats = [self.degree2quat(*x) for x in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]]

        points = []
        i = 0

        all_instance_feature_dict = defaultdict(list)
        all_instance_feature_dict_final = dict()

        for x in tqdm([x for x in grid_points if self.pathfinder.is_navigable(x)]):
            # if i > 10: continue
            if not self.inside_p(x, room_bbox): continue
            new_state = habitat_sim.AgentState(x, [0, 0, 0, 1])
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

            if return_features:
                instance_feature_dict = self.get_per_instance_feature(i, bbox_file.replace(".json", ""), obs)
                for instance, feature in instance_feature_dict.items():
                    all_instance_feature_dict[instance].append(feature)

            i += 6

        return all_instance_feature_dict_final

    def get_per_instance_feature(self, i, room, obs):
        instance_feature_dict = dict()

        for (j,frame) in enumerate(obs):
            image = frame[..., :3].astype(np.uint8)  
            image_features = self.encoder.encode(image)
            pil_image = Image.fromarray(image)


            all_semantics = frame[..., 4]
            semantics = np.unique(all_semantics).astype(int)

            for semantic in semantics:
                # if semantic < 10000: continue
                indices = np.where(all_semantics == semantic)
                if indices[0].shape[0] < 10: continue
                ymin, ymax, xmin, xmax = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
                image_copy = copy.deepcopy(image)
                image_copy[all_semantics != semantic] = 255
                pil_image = Image.fromarray(image_copy)
                
                # 

                cropped_image = pil_image.crop((xmin-1, ymin-1, xmax+1, ymax+1))
                cropped_image = np.array(cropped_image)

                pil_image.save("./tmp/%s/%d_%d.jpg"%(room, semantic, i+j))

                cropped_features = self.encoder.encode(cropped_image)
                instance_feature_dict[semantic] = cropped_features.mean(1).detach().cpu().numpy()

        return instance_feature_dict

    def parse_visual_observation(self, obs):
        rgb = cv2.cvtColor(obs["rgba"], cv2.COLOR_RGBA2RGB)
        frame = np.concatenate([rgb, obs["depth"][:, :, np.newaxis], obs["semantic"][:, :, np.newaxis]], axis=-1)

        return frame

    def get_semantic_labels(self):
        id2cate = dict()
        if self.new_objs is not None:
            for i in self.new_objs:
                id2cate[i["semantic_id"]] = i["cate"]
        with open(os.path.join(config.HM3D_DIR, _scene, f"{_scene.split('-')[1]}.semantic.txt"), "r") as f:
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


if __name__ == "__main__":
    objaverse = Objaverse(selected=False)
    objectfolder = ObjectFolder()
    bbox_dir = config.BBOX_WITH_ADDED_OBJECTS_DIR

    # for bbox_file in os.listdir(bbox_dir):
    for bbox_file in ["00009-vLpv2VX547B_7.json"]:
        print ("Processing %s"%bbox_file)
        bboxes = json.load(open(os.path.join(bbox_dir, bbox_file)))["incremented_bboxes"]
        new_objs = []
        objectfolder_cats = []
        scene = bbox_file.split("_")[0]
        objaverse_dict = dict()
        for bbox in bboxes:

            if "source" in bbox and bbox["source"] == "objaverse":
                class_name = bbox["class_name"].replace("(soft)", "").replace("(hard)", "").replace("(deformable)", "").replace("(not deformable)", "").strip()
                
                try:    
                    if class_name in objaverse_dict: id2 = objaverse_dict[class_name]
                    else: id2 = random.choice(objaverse.lvis[class_name]); objaverse_dict[class_name] = id2
                    path = objaverse.get_objects([id2])[id2]
                    new_obj = {"path": path, "bbox": bbox["bbox"], "id": bbox["id"]}
                    new_objs.append(new_obj)                 
                except:
                    continue

        encoder = LlaVa_Encoder()
        sampler = GridSampler(scene, new_objs, audio=False, encoder=encoder)
        print ("successfully building sampler")

        room = bbox_file.replace(".json", "").split("_")[1]
        room_bbox = json.load(open(os.path.join(config.ROOM_BBOX_DIR, bbox_file.split("_")[0]+".json")))[room]
        room_bbox = [[room_bbox[0][0], room_bbox[0][2], room_bbox[0][1]], [room_bbox[1][0], room_bbox[1][2], room_bbox[1][1]]]

        feature_dict = sampler.scan_scene(bbox_file, return_features=True)
        
        print (feature_dict)
