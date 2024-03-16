import quaternion # Remove this will cause invalid pointer error !!!!
import habitat_sim
import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.sims.habitat_simulator.actions import HabitatSimActions

import os
import librosa
import numpy as np
import magnum as mn
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from moviepy.editor import VideoFileClip
from habitat_sim.utils import viz_utils as vut
from moviepy.audio.AudioClip import AudioArrayClip
from utils import config
from typing import List

os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"


class MultisensorySimulator(habitat_sim.Simulator):
    def __init__(self, conf: habitat_sim.Configuration, new_objs: List[dict] = None):
        super().__init__(conf)
        # Fake habitat.core.simulator.Simulator
        self._sim = self
        # self.habitat_config = habitat.get_config()
        # self.habitat_config["SCENE"] = conf.sim_cfg.scene_id

        # Assign semantic ids & place new objs
        self.new_objs = new_objs
        if self.new_objs is not None:
            count = 0
            for i in self.new_objs:
                _id = 10000 + count
                i["semantic_id"] = _id
                count += 1
            self._place_objs()

        # Others
        assert len(self.agents) == 1
        
        # TODO: change this
        self.observations = [self.get_sensor_observations()] # Init observation after audio setup

    def _place_objs(self):
        obj_attr_mgr = self.get_object_template_manager()
        rigid_obj_mgr = self.get_rigid_object_manager()

        for i in self.new_objs: # v in (x, y, z) format but hm3d in (x, z, y) format
            k = i["path"]
            v = i["bbox"]
            # Calc scale
            object_template = obj_attr_mgr.create_new_template(k)
            obj_temp_id = obj_attr_mgr.register_template(object_template)
            obj = rigid_obj_mgr.add_object_by_template_id(obj_temp_id)
            _bbox = obj.root_scene_node.compute_cumulative_bb()
            _scale = (v[1][2] - v[0][2]) / (_bbox.top - _bbox.bottom)
            rigid_obj_mgr.remove_object_by_id(obj.object_id)
            obj_attr_mgr.remove_template_by_id(obj_temp_id)

            # Add new mesh
            object_template.scale = np.ones(3) * _scale
            object_template.semantic_id = i["semantic_id"]
            obj_temp_id = obj_attr_mgr.register_template(object_template)
            obj = rigid_obj_mgr.add_object_by_template_id(obj_temp_id)
            i["obj_id"] = obj.object_id

            # Move object
            _loc = [(v[0][0] + v[1][0]) / 2, v[0][2], (v[0][1] + v[1][1]) / 2]

            _bbox = obj.root_scene_node.compute_cumulative_bb()
            obj.translation = -_bbox.center() + _loc
            if "rot" in i:
                obj.rotation = mn.Quaternion.rotation(mn.Deg(i["rot"]), [0.0, 1.0, 0.0])
            if "mass" in i:
                obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                obj.mass = i["mass"]
            else:
                obj.motion_type = habitat_sim.physics.MotionType.STATIC

            print(i["cate"], obj.translation, _scale)

        # TODO: enable
        # self.update_navmesh()

    def update_navmesh(self):
        # # recompute the NavMesh with STATIC objects
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.include_static_objects = True
        navmesh_success = self.recompute_navmesh(self.pathfinder, navmesh_settings)
        if not navmesh_success:
            raise Exception("Recompute Navmesh Fail.")

    def set_audio_source(self, loc):
        audio_sensor = self.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioSourceTransform(loc)

    def show_top_down_map(self, meters_per_pixel=0.1, height=0., path_points=None):
        print(f"The NavMesh bounds in {height} are: " + str(self.pathfinder.get_bounds()))
        top_down_map = self.pathfinder.get_topdown_view(meters_per_pixel, height)
        top_down_map = 1. - top_down_map * 0.5 # Recolor

        plt.figure(figsize=(12, 8))
        plt.axis("off")
        plt.imshow(top_down_map, cmap='gray', vmin=0., vmax=1.)

        top_down_loc = self._convert_points_to_topdown([self.agent_loc], meters_per_pixel)[0]
        plt.plot(*top_down_loc, marker="o", markersize=10, alpha=0.8)

        if path_points:
            top_down_loc = self._convert_points_to_topdown(path_points, meters_per_pixel)
            plt.plot(*np.array(top_down_loc).transpose(), marker="o", markersize=5, alpha=0.8)
        plt.show()

    def _convert_points_to_topdown(self, points, meters_per_pixel):
        bounds = self.pathfinder.get_bounds()

        # convert 3D x,z to topdown x,y
        points_topdown = []
        for point in points:
            px = (point[0] - bounds[0][0]) / meters_per_pixel
            py = (point[2] - bounds[0][2]) / meters_per_pixel
            points_topdown.append(np.array([px, py]))
        return points_topdown

    def _path_planning(self, target_loc):
        path = habitat_sim.ShortestPath()
        path.requested_start = self.agent_loc
        path.requested_end = target_loc
        found_path = self.pathfinder.find_path(path)
        if found_path:
            return path.points
        else:
            return [self.agent_loc]

    def move_agent_to_target(self, target_loc, goal_radius=1., final_goal_radius=0.):
        # First point is current position. Last point is target location.
        path_points = self._path_planning(target_loc)
        print(f"Move path {path_points}")

        shortest_path_follower = ShortestPathFollower(sim=self, goal_radius=goal_radius, return_one_hot=False)
        for idx, i in enumerate(path_points):
            if (idx + 1) == len(path_points):
                shortest_path_follower = ShortestPathFollower(sim=self, goal_radius=final_goal_radius, return_one_hot=False)
            while True:
                next_action = shortest_path_follower.get_next_action(i)
                if next_action == HabitatSimActions.stop:
                    break
                elif next_action == HabitatSimActions.move_forward:
                    action = "move_forward"
                elif next_action == HabitatSimActions.turn_left:
                    action = "turn_left"
                elif next_action == HabitatSimActions.turn_right:
                    action = "turn_right"
                else:
                    raise Exception(f"Action {next_action} not defined.")

                assert action in self.agent_actions.keys()
                obs = self.step(action)
                self.observations.append(obs)

    def calculate_audio(self):
        rirs = [np.array(x["audio_sensor"]).T for x in self.observations]
        audio_data, _ = librosa.load(self.audio_objs[0].audio_path, sr=config.RIR_SAMPLING_RATE)

        index = 0
        audio = []
        # TODO: change this
        scaled_sample_rate = int(config.RIR_SAMPLING_RATE * self.step_delta_t)
        for i in rirs:
            if index * scaled_sample_rate - i.shape[0] < 0:
                source_sound = audio_data[: (index + 1) * scaled_sample_rate]
                binaural_convolved = np.array([fftconvolve(source_sound, i[:, channel]) for channel in range(i.shape[-1])])
                audio_goal = binaural_convolved[:, index * scaled_sample_rate: (index + 1) * scaled_sample_rate]
            else:
                # include reverb from previous time step
                source_sound = audio_data[index * scaled_sample_rate - i.shape[0] + 1: (index + 1) * scaled_sample_rate]
                binaural_convolved = np.array([fftconvolve(source_sound, i[:, channel], mode='valid') for channel in range(i.shape[-1])])
                audio_goal = binaural_convolved
            audio.append(audio_goal)
            index = (index + 1) % (audio_data.shape[0] // scaled_sample_rate)
        return np.concatenate(audio, axis=-1).transpose()

    def set_material_file(self, sensor_key, file_path):
        self.agents[0]._sensors[sensor_key].setAudioMaterialsJSON(file_path)

    @property
    def agent_loc(self):
        return self.agents[0].state.position

    @property
    def agent_rot(self):
        return self.agents[0].state.rotation

    @property
    def agent_actions(self):
        return dict(self.agents[0].agent_config.action_space)


# TODO: move all audio related code into fake sim
def demo(sim, fps, target=None):
    # Navigation and first-person video
    if target is None:
        target = sim.pathfinder.get_random_navigable_point()

    sim.move_agent_to_target(target)
    vut.make_video(sim.observations, "rgba", "color", "../color.mp4", fps=fps, open_vid=False)
    vut.make_video(sim.observations, "depth", "depth", "../depth.mp4", fps=fps, open_vid=False)
    vut.make_video(sim.observations, "semantic", "semantic", "../semantic.mp4", fps=fps, open_vid=False)

    # Calculate audio and merge with video
    # audio = sim.calculate_audio()
    # _audio = AudioArrayClip(audio, fps=RIR_SAMPLING_RATE)
    # _audio.write_audiofile("./demo.wav")

    # _video = VideoFileClip("./tmp.mp4")
    # _video = _video.set_audio(_audio)
    # _video.write_videofile("./demo.mp4")
    # os.remove("tmp.mp4")
