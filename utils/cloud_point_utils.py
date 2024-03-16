import math
import random
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class Reconstruct3D(object):
    def __init__(self, h, w, hfov, camera2agent):
        self.h = h
        self.w = w
        self.focal_length = (w / 2) / math.tan(np.deg2rad(hfov / 2))
        self.camera2agent = camera2agent
        self.voxel_size = 0.5
        self.num_points_per_voxel = 1000

    def depth_map2points(self, depth_map, quat, loc):
        rot = Rotation.from_quat(quat)

        # Depth to agent coordinate
        _max = 100
        _min = 0
        valid_mask = (depth_map > _min) & (depth_map < _max)
        depth_map = np.clip(depth_map, _min, _max)

        _x, _z = np.meshgrid(np.arange(self.w), np.arange(self.h - 1, -1, -1))
        x = (_x - (self.w - 1) / 2.) * depth_map / self.focal_length
        y = depth_map
        z = (_z - (self.h - 1) / 2.) * depth_map / self.focal_length
        _points = np.stack([x, z, y], axis=-1).reshape(-1, 3)
        # Rotate points
        _points = rot.inv().apply(_points)
        # Agent to world coordinate
        _points[:, 0] += loc[0]
        _points[:, 1] += loc[1]
        _points[:, 2] = loc[2] - _points[:, 2]  # reverse axis
        return _points + self.camera2agent, valid_mask.reshape(-1)

    def downsample_index(self, points):
        # Drop far points
        dist = np.linalg.norm(points, axis=1)
        valid_idx = np.where(dist < 100)[0]
        valid_points = points[valid_idx, :]

        # Build voxels
        min_coord = np.array([np.min(valid_points[:, 0]), np.min(valid_points[:, 1]), np.min(valid_points[:, 2])])
        max_coord = np.array([np.max(valid_points[:, 0]), np.max(valid_points[:, 1]), np.max(valid_points[:, 2])])
        num_voxels_x = int((max_coord[0] - min_coord[0]) / self.voxel_size) + 1
        num_voxels_y = int((max_coord[1] - min_coord[1]) / self.voxel_size) + 1
        num_voxels_z = int((max_coord[2] - min_coord[2]) / self.voxel_size) + 1
        print(len(points), num_voxels_x, num_voxels_y, num_voxels_z)
        voxel_grid = np.zeros((num_voxels_x, num_voxels_y, num_voxels_z), dtype=object)
        for i in range(num_voxels_x):
            for j in range(num_voxels_y):
                for k in range(num_voxels_z):
                    voxel_grid[i, j, k] = []

        # Assign points to voxels
        voxel_indices = ((valid_points - min_coord) / self.voxel_size).astype(int)
        for i, idx in tqdm(zip(valid_idx, voxel_indices)):
            voxel_grid[idx[0], idx[1], idx[2]].append(i)

        # Random sampling in voxels
        res = []
        for i in tqdm(voxel_grid.flatten()):
            if len(i) > self.num_points_per_voxel:
                res.extend(np.random.choice(i, self.num_points_per_voxel, replace=False))
            else:
                res.extend(i)
        return res


def crop_points(points, bboxes):
    coords = points[:, :3]
    idxs = None
    for i in bboxes:
        idx = np.all((coords < i[1]) & (coords > i[0]), axis=1)
        if idxs is None:
            idxs = idx
        else:
            idxs = (idxs | idx)
    points = points[idxs, :]
    return points
