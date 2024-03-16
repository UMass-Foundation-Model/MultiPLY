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
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from PIL import Image
import copy
from collections import defaultdict
import random
from utils.cloud_point_utils import Reconstruct3D, crop_points

class GridSampler(MultisensorySimulator):
    def __init__(self, scene, new_objs=None, audio=False):
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
        # self.encoder = encoder

    @staticmethod
    def inside_p(pt, box):
        if pt[0] < box[0][0] or pt[0] > box[1][0]: return False
        if pt[1] < box[0][1] or pt[1] > box[1][1]: return False
        if pt[2] < box[0][2] or pt[2] > box[1][2]: return False
        if box[1][0] - pt[0] < 0 or pt[0] - box[0][0] < 0: return False
        if box[1][2] - pt[2] < 0 or pt[2] - box[0][2] < 0: return False
        return True

    def scan_scene(self, bbox_file, all_ids, scene, return_features = True):
        room_bbox = bbox_file

        grid_points = np.load(os.path.join(config.SAMPLE_DIR, self.scene, "grid_points.npy"))


        grid_points = crop_points(grid_points, [room_bbox])
        quats = [self.degree2quat(*x) for x in [(0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90)]]

        points = []
        i = 0

        all_instance_feature_dict = defaultdict(list)
        all_instance_feature_dict_final = dict()

        for x in [x for x in grid_points if self.pathfinder.is_navigable(x)]:    
            new_state = habitat_sim.AgentState(x, [0, 0, 0, 1])
            self.agents[0].set_state(new_state) # set position

            # Scan in sphere
            obs = [self.parse_visual_observation(self.get_sensor_observations()),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left")),
                   self.parse_visual_observation(self.step("look_left"))]


            self.get_per_instance(i, scene.replace(".json", ""), obs, all_ids)

            i += 6

            # convert to points
            coordinates = []
            valid_masks = []
            for o, q in zip(obs, quats):
                p, valid_mask = self.reconstructor.depth_map2points(o[:, :, 3], q, x)
                coordinates.append(p)
                valid_masks.append(valid_mask)
            coordinates = np.concatenate(coordinates, axis=0)
            valid_idx = np.where(np.concatenate(valid_masks, axis=0))[0]

            # Concat all
            obs = np.stack(obs, axis=0).reshape(-1, 5)
            _points = np.concatenate([coordinates, obs[:, :3], obs[:, 4:]], axis=1).astype(np.float16) # xzy, rgb, semantic
            _points = _points[valid_idx]
            points.append(_points)
        

        if not len(points): return np.zeros((1,1))

        print ("%d vertices inside the room"%len(points))
        points = np.concatenate(points, axis=0)
        print ("%d points inside the room"%points.shape[0])
        points = crop_points(points, [room_bbox])
        print ("%d points after crop"%points.shape[0])
        if points.shape[0] == 0: return np.zeros((1,1))

        idx = self.reconstructor.downsample_index(points[:, :3])
        points = points[idx, :]
        print ("%d points after sampling"%points.shape[0])
        
        return points

    def get_per_instance(self, i, room, obs, all_ids):
        instance_feature_dict = dict()

        for (j,frame) in enumerate(obs):
            image = frame[..., :3].astype(np.uint8)  
            depth = frame[..., 3].astype(np.uint8)  

            pil_image = Image.fromarray(image)

            try:
                os.mkdir("./data/original_2d_gt_seg/%s"%room)
            except:
                pass

            np.save("./data/original_2d_gt_seg/%s/depth_%d.npy"%(room, i+j), depth)
            pil_image.save("./data/original_2d_gt_seg/%s/image_%d.jpg"%(room, i+j))

            all_semantics = frame[..., 4]
            semantics = np.unique(all_semantics).astype(int)

            for semantic in semantics:
                if not (semantic in all_ids or str(semantic) in all_ids or semantic >= 10000): continue
                
                indices = np.where(all_semantics == semantic)
                if np.min(indices[0]) == 0 or np.min(indices[1]) == 0 or np.max(indices[1]) == 719 or np.max(indices[0]) == 719: continue
                if indices[0].shape[0] < 100: continue 
                
                ymin, ymax, xmin, xmax = np.min(indices[0]), np.max(indices[0]), np.min(indices[1]), np.max(indices[1])
                image_copy = copy.deepcopy(image)
                image_copy[all_semantics != semantic] = 255
                pil_image = Image.fromarray(image_copy)
                
                pil_image.save("./data/original_2d_gt_seg/%s/%d_%d.jpg"%(room, semantic, i+j))

                cropped_image = pil_image.crop((xmin-1, ymin-1, xmax+1, ymax+1))

                cropped_image.save("./data/original_2d_gt_seg/%s/%d_%d_cropped.jpg"%(room, semantic, i+j))

    # Add features
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

    bbox_dir = config.HM3D_BBOX_DIR
    hm3d = HM3D()

    for bbox_file in tqdm(os.listdir(bbox_dir)):

        print ("Processing %s"%bbox_file)
        bboxes = json.load(open(os.path.join(bbox_dir, bbox_file)))
 
        new_objs = []
        objectfolder_cats = []
        scene = bbox_file.split("_")[0]
        objaverse_dict = dict()

        final_bboxes = []
        original_bboxes = []
        all_ids = []

        bbox_file_copy = bbox_file
        scene2 = bbox_file.split("_")[0]+".json"
        room2 = bbox_file.replace(".json", "").split("_")[1]
        try:
            room_bbox = json.load(open(os.path.join(config.ROOM_BBOX_DIR, scene2)))[room2]
        except:
            continue

        k = 1

        for bbox in bboxes:
            if "source" in bbox and bbox["source"] == "objectfolder":
                try:
                    cat, id2, material, path = objectfolder.get_objects(bbox["class_name"].replace("_hot", "").replace("_cold", ""))
                    new_obj = {"path": path, "bbox": bbox["bbox"], "id": bbox["id"]}
                    new_objs.append(new_obj)
                    final_bboxes.append(bbox)
                except:
                    continue

            elif "source" in bbox and bbox["source"] == "objaverse":
                class_name = bbox["class_name"].replace("_hard", "").replace("_soft", "").replace("_hot", "").replace("_cold", "").strip()
                
                try:    
                    id2 = random.choice(objaverse.lvis[class_name])
                    path = objaverse.get_objects([id2])[id2]
                    new_obj = {"path": path, "bbox": bbox["bbox"], "id": bbox["id"]}
                    new_objs.append(new_obj) 
                    final_bboxes.append(bbox)                
                except:
                    continue

            else:
                if not bbox['class_name'] in ['floor', 'wall', 'ceiling']:
                    all_ids.append(bbox['id'])
                    final_bboxes.append(bbox)

                original_bboxes.append(bbox)
            
            final_bboxes.append(bbox)
        

        sampler = GridSampler(scene, new_objs, audio=False)
        print ("successfully building sampler")

        points = sampler.scan_scene(room_bbox, all_ids, bbox_file_copy, return_features=True)
        sampler.close()

        if points.shape[0] == 1: continue

        np.save(os.path.join(config.SAMPLE_DIR, scene, f"{bbox_file_copy}.npy"), points)
