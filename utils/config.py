import os.path
from pathlib import Path


# Dir
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
HM3D_DIR = os.path.join(DATA_DIR, "hm3d")
HM3D_BBOX_DIR = os.path.join(DATA_DIR, "new_hm3d_obj_bbox")
OBJAVERSE_DIR = os.path.join(DATA_DIR, "objaverse")
AUDIOSET_DIR = os.path.join(DATA_DIR, "audio_set")
TASK_TEMPLATE_DIR = os.path.join(DATA_DIR, "task_template")
SAMPLE_DIR = os.path.join(DATA_DIR, "sampled_data")
OBJECTFOLDER_DIR = os.path.join(DATA_DIR, "object_folder")
OBJECTFOLDER_OBJECTS_DIR = os.path.join(DATA_DIR, "ObjectFolder")
BBOX_WITH_ADDED_OBJECTS_DIR = os.path.join(DATA_DIR, "bbox_with_added_objects")
BBOX_WITH_ADDED_OBJECTS_DIR2 = os.path.join(DATA_DIR, "bbox_with_added_objects2")
BBOX_WITH_ADDED_OBJECTFOLDER_DIR = os.path.join(DATA_DIR, "bbox_with_added_objectfolder")
BBOX_WITH_ADDED_OBJECTFOLDER_DIR2 = os.path.join(DATA_DIR, "bbox_with_added_objectfolder2")
BBOX_WITH_TEMPERATURE_DIR = os.path.join(DATA_DIR, "bbox_with_temperature")
BBOX_WITH_TEMPERATURE_DIR2 = os.path.join(DATA_DIR, "bbox_with_temperature2")
ROOM_BBOX_DIR = os.path.join(DATA_DIR, "room_bboxes_revised_axis")
THIRD_PARTY_DIR = os.path.join(ROOT_DIR, "third_party")
AUDIO_EMBEDDING_DIR = os.path.join(ROOT_DIR, "embedding")

# GPT
OPENAI_KEY = ""
# OPENAI_PROXY = {"http": "127.0.0.1:7890", "https": "127.0.0.1:7890"}
OPENAI_PROXY = {}

# Simulator
RIR_SAMPLING_RATE = 16000
def sim_conf(scene: str, visual=True, audio=True, physics=False):
    import quaternion # Remove this will cause invalid pointer error !!!!
    import habitat_sim
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = os.path.join(HM3D_DIR, scene, f"{scene.split('-')[1]}.basis.glb")
    # TODO: change this
    backend_cfg.scene_dataset_config_file = os.path.join(HM3D_DIR, "hm3d_annotated_train_basis.scene_dataset_config.json")
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = physics

    sensors = []
    if visual:
        camera_resolution = [720, 720] # h = w for scene scan
        camera_position = [0.0, 1.4, 0.0]
        _spec = habitat_sim.CameraSensorSpec()
        _spec.uuid = "rgba"
        _spec.sensor_type = habitat_sim.SensorType.COLOR
        _spec.resolution = camera_resolution
        _spec.position = camera_position
        _spec.orientation = [0.0, 0.0, 0.0]
        _spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors.append(_spec)

        _spec = habitat_sim.CameraSensorSpec()
        _spec.uuid = "depth"
        _spec.sensor_type = habitat_sim.SensorType.DEPTH # COLOR = 1, DEPTH = 2, SEMANTIC = 4
        _spec.resolution = camera_resolution
        _spec.position = camera_position
        _spec.orientation = [0.0, 0.0, 0.0]
        _spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors.append(_spec)

        _spec = habitat_sim.CameraSensorSpec()
        _spec.uuid = "semantic"
        _spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        _spec.resolution = camera_resolution
        _spec.position = camera_position
        _spec.orientation = [0.0, 0.0, 0.0]
        _spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensors.append(_spec)

    if audio:
        _spec = habitat_sim.AudioSensorSpec()
        _spec.uuid = "audio_sensor" # Must use this name or backend simulator will raise error :(
        _spec.enableMaterials = False
        _spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        _spec.channelLayout.channelCount = 2
        _spec.acousticsConfig.sampleRate = RIR_SAMPLING_RATE
        _spec.acousticsConfig.indirect = True
        sensors.append(_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensors
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    return cfg