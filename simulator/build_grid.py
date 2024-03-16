import os
import quaternion # Remove this will cause invalid pointer error !!!!
import habitat_sim
import numpy as np
from utils import config


class GridBuilder(habitat_sim.Simulator):
    def __init__(self, scene):
        self.scene = scene
        backend_cfg = habitat_sim.SimulatorConfiguration()
        
        backend_cfg.scene_id = os.path.join(config.HM3D_DIR, scene, f"{scene.split('-')[1]}.basis.glb")
        # TODO: change this
        backend_cfg.scene_dataset_config_file = os.path.join(config.HM3D_DIR, "hm3d_annotated_train_basis.scene_dataset_config.json")
        backend_cfg.load_semantic_mesh = True
        backend_cfg.enable_physics = False
        cfg = habitat_sim.Configuration(backend_cfg, [habitat_sim.agent.AgentConfiguration()])
        super().__init__(cfg)

    def build_grids_if_not_exist(self):
        _num = self.pathfinder.num_islands
        _area = [self.pathfinder.island_area(x) for x in range(_num)]
        _idx = _area.index(max(_area)) # Assert only one largest island
        vertices = self.pathfinder.build_navmesh_vertices(_idx)
        unique_vertices = np.unique(vertices, axis=0)

        save_path = os.path.join(config.SAMPLE_DIR, self.scene)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "grid_points.npy"), unique_vertices)
        # Nearest point in cube (cKDTree)


if __name__ == "__main__":
    print("Switch to latest version of habitat before running the code! Or Error")
    folder_list = [folder for folder in  os.listdir(config.HM3D_DIR) if folder.startswith("00") and len(os.listdir(os.path.join(config.HM3D_DIR, folder))) == 4]

    for i in os.listdir(config.HM3D_DIR):
        if os.path.isdir(os.path.join(config.HM3D_DIR, i)):
            sim = GridBuilder(i)
            sim.build_grids_if_not_exist()
