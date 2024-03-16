import copy
import os
import librosa
import random
import numpy as np
import pandas as pd
import json
from typing import List
import itertools
from collections import defaultdict
from utils import config
import re
from tqdm import tqdm
import objaverse
from collections import defaultdict

class HM3D(object):
    def __init__(self):
        self.dir_path = config.HM3D_BBOX_DIR
        self.augmented_dir_path = config.BBOX_WITH_ADDED_OBJECTS_DIR
        self.objectfolder_dir_path = config.BBOX_WITH_ADDED_OBJECTFOLDER_DIR
        self.scene2cate = dict()
        self.files = sorted(x for x in os.listdir(self.dir_path) if len(self.load_room(x)) > 0)
        self.augmented_files = sorted(x for x in os.listdir(self.augmented_dir_path) if len(self.load_room_with_added_objects(x)) > 0)
        self.objectfolder_files = sorted(x for x in os.listdir(self.objectfolder_dir_path) if len(self.load_room_with_added_objectfolder(x)) > 0)

        for i in self.files:
            scene = i.split("_")[0]
            if scene not in self.scene2cate:
                self.scene2cate[scene] = []

            bboxes = json.load(open(os.path.join(self.dir_path, i), "r"))
            for b in bboxes:
                self.scene2cate[scene].append(b["class_name"])
        self.categories = list(sorted(set(itertools.chain.from_iterable(self.scene2cate.values()))))
        self.scenes = list(sorted(set(self.scene2cate.keys())))

    def load_room(self, room_json: str):
        # Coordinates in (x,y,z) format
        bboxes = json.load(open(os.path.join(self.dir_path, room_json), "r"))
        return bboxes

    def load_room_with_added_objects(self, room_json: str):
        # Coordinates in (x,y,z) format
        bboxes = json.load(open(os.path.join(self.augmented_dir_path, room_json), "r"))
        return bboxes

    def load_room_with_added_objectfolder(self, room_json: str):
        # Coordinates in (x,y,z) format
        bboxes = json.load(open(os.path.join(self.objectfolder_dir_path, room_json), "r"))
        return bboxes

    def load_scene(self, scene: str):
        bboxes = []
        for i in self.files:
            if scene in i:
                _b = self.load_room(i)
                bboxes.extend(_b)
        return bboxes

    @staticmethod
    def room_bbox(bboxes):
        _min = np.round(np.min(np.array([x["bbox"][0] for x in bboxes]), axis=0), 3)
        _max = np.round(np.max(np.array([x["bbox"][1] for x in bboxes]), axis=0), 3)
        return list(_min), list(_max)

    def scene_description(self, room_json: str, max_box=50, format="topbottom", add_id = False):
        bboxes = self.load_room(room_json)
        if len(bboxes) > max_box:
            bboxes = random.choices(bboxes, k=max_box)
        np.set_printoptions(suppress=True)
        
        if format == "topbottom":
            _min, _max = self.room_bbox(bboxes)
            room_desc = f"<room>: [{_min}, {_max}]\n"
            if add_id:
                obj_desc = "\n".join([f'<{x["class_name"]}>({x["id"]}): {np.round(x["bbox"], 3)}'.replace("\n", ",") for x in bboxes])
            else:
                obj_desc = "\n".join([f'{x["class_name"]}: {np.round(x["bbox"], 3)}'.replace("\n", ",") for x in bboxes])
            return room_desc + obj_desc
        elif format == "center":
            obj_desc = "\n".join([f'<{x["class_name"]}>({x["id"]}): {np.round((np.array(x["bbox"][0]) + np.array(x["bbox"][1])) / 2, 3)}'.replace("\n", ",") for x in bboxes])
            return obj_desc
        elif format == "object_name":
            obj_desc = "\n".join([f'<{x["class_name"]}>({x["id"]})'.replace("\n", ",") for x in bboxes])
            return obj_desc

    def multi_modal_scene_description(self, room_json: str, max_box=50):
        try:
            bboxes = self.load_room_with_added_objects(room_json)['incremented_bboxes']
        except:
            bboxes = self.load_room_with_added_objectfolder(room_json)['incremented_bboxes']
        bboxes = bboxes[:max_box]
        np.set_printoptions(suppress=True)

        final_bboxes = []
        for bbox in bboxes:
            if "source" in bbox and bbox["source"] == "objaverse":
                bbox["class_name"] = bbox["class_name"] + "(audio, tactile)"
            elif "source" in bbox and bbox["source"] == "objectfolder":
                bbox["class_name"] = bbox["class_name"] + "(tapsound)"
            new_bbox_format = f'<{bbox["class_name"]}>({bbox["id"]}): {np.round(bbox["bbox"], 3)}'.replace("\n", ",")

            final_bboxes.append(new_bbox_format)

        obj_desc = "\n".join(final_bboxes)
        return obj_desc


# TODO: include evaluation set
class AudioSet(object):
    def __init__(self, training_set=True):
        self.dir_path = config.AUDIOSET_DIR

        # Load
        ontology = json.load(open(os.path.join(self.dir_path, "ontology.json"), "r"))
        if training_set:
            meta_file = "unbalanced_train_segments.csv"
        else:
            meta_file = "eval_segments.csv"
        meta = pd.read_csv(os.path.join(self.dir_path, meta_file), sep=", ", engine='python', skiprows=2)

        # Set selected categories
        # Ontology is a graph, not a tree. query_handles is visible tree roots.
        query_handles = ["Music", "Sounds of things"]
        valid_cate = ["Musical instrument", "Domestic sounds, home sounds", "Liquid", "Glass", "Printer",
                      "Air conditioning", "Mechanical fan", "Clock", "Fire alarm", "Smoke detector, smoke alarm",
                      "Doorbell", "Alarm clock", "Ringtone", "Telephone bell ringing", "Domestic sounds, home sounds",
                      "Loudspeaker", "Radio", "Television", "MP3", "Domestic animals, pets"]
        block_cate = ["Human sounds", "Vehicle"]

        # Put audios on the node.
        name2node = {x["name"]: x["id"] for x in ontology}
        node2child = {x["id"]: x["child_ids"] for x in ontology}
        valid_nodes = self.iterative_query([name2node[x] for x in valid_cate], query_dict=node2child)
        block_nodes = self.iterative_query([name2node[x] for x in block_cate], query_dict=node2child)

        self._node2audio = defaultdict(list)
        for id, labels in zip(meta["# YTID"], meta["positive_labels"]):
            labels = set(labels.strip('"').split(","))
            if len(labels & block_nodes): continue
            for i in (labels & valid_nodes):
                self._node2audio[i].append(id)

        # Pruning nodes without audios
        self.nodes = list()
        for i in [x["id"] for x in ontology]:
            nodes = self.iterative_query([i], node2child)
            if any(len(self._node2audio[x]) for x in nodes):
                self.nodes.append(i)

        query_nodes = self.iterative_query([name2node[x] for x in query_handles], query_dict=node2child)
        self.nodes = list(set(self.nodes) & query_nodes)

        filtered_ontology = [x for x in ontology if x["id"] in self.nodes]
        self.node2name = {x["id"]: x["name"] for x in filtered_ontology}
        self.node2description = {x["id"]: x["description"] for x in filtered_ontology}
        self.node2child = {x["id"]: (set(x["child_ids"]) & set(self.nodes)) for x in filtered_ontology}
        self.node2father = defaultdict(list)
        for k, v in self.node2child.items():
            for i in v:
                self.node2father[i].append(k)

        # Others
        self.audio_ids = self.get_ids(self.nodes)
        self.meta = meta[meta["# YTID"].isin(self.audio_ids)]
        self.downloader = os.path.join(config.THIRD_PARTY_DIR, "youtube-dl")
        print(f"AudioSet {meta_file}: {len(self.meta)} / {len(meta)}, cate {len(self.nodes)} / {len(ontology)}")


        # Display
        root_nodes = set(self.nodes).difference(set(itertools.chain.from_iterable(self.node2child.values())))
        self.print_tree(root_nodes)

    def get_ids(self, nodes: List[str]):
        nodes = self.iterative_query(nodes, self.node2child)
        return list(set(itertools.chain.from_iterable(self._node2audio[x] for x in nodes)))

    def get_audio(self, audio_id):
        assert audio_id in self.audio_ids # YTID is unique in training set
        info = self.meta[self.meta["# YTID"] == audio_id]
        assert len(info) == 1
        _path = os.path.join(config.AUDIOSET_DIR, f"{audio_id}.wav")
        if not os.path.exists(_path):
            os.system(f"sh {os.path.join(config.THIRD_PARTY_DIR, 'fetch_audio.sh')} "
                      f"{audio_id} {info['start_seconds'].values[0]} {info['end_seconds'].values[0]} "
                      f"{_path} {self.downloader}")

        audio_data = None
        success = False
        if os.path.exists(_path):
            audio_data, _ = librosa.load(_path, sr=config.RIR_SAMPLING_RATE)
            success = True
        return audio_data, success

    # @staticmethod
    # def iterative_query(nodes: List[str], query_dict: dict[str, List[str]], include_root=True) -> set:
    #     q = copy.deepcopy(nodes)
    #     res = []
    #     while len(q):
    #         node = q.pop()
    #         res.append(node)
    #         q.extend(query_dict[node])

    #     if not include_root:
    #         for i in nodes:
    #             res.remove(i)
    #     return set(res)

    def print_tree(self, nodes, max_depth=100):
        q = []
        for i in nodes:
            q.append((i, 0))
        while len(q):
            node, depth = q.pop()
            for i in self.node2child[node]:
                q.append((i, depth+1))
            if depth < max_depth:
                num = len(set(itertools.chain.from_iterable(
                    self._node2audio[x] for x in self.iterative_query([node], self.node2child))))
                print(f'{"--" * depth} {self.node2name[node]}: {num}, {self.node2description[node]}')

    @property
    def meta_info(self):
        info = {}
        for i in self.nodes:
            path = self.iterative_query([i], self.node2father)
            tags = [self.node2name[x] for x in path]
            description = self.node2description[i]
            info[self.node2name[i]] = f"tags={tags}, description='{description}'"
        return info


class Objaverse(object):
    def __init__(self, selected=True):
        self.dir_path = config.OBJAVERSE_DIR
        if not selected:
            with open(os.path.join(config.OBJAVERSE_DIR, "audio_objaverse.txt"), "r") as f:
                valid_cate = f.readlines()
        else:
            with open(os.path.join(config.OBJAVERSE_DIR, "selected_objaverse_lvis.txt"), "r") as f:
                valid_cate = f.readlines()
        valid_cate = [x.strip("\n") for x in valid_cate]
        self.lvis = {k.strip(): v for k, v in objaverse.load_lvis_annotations().items() if k in valid_cate}
        self.categories = sorted(valid_cate)

        # self.meta_info = []
        # for k, v in self.anns.items():
        #     info = {'id': k}
        #
        #     if len(v["name"]):
        #         info["name"] = v["name"]
        #     if k in self.lvis: # Precise labels
        #         info["label"] = self.lvis[k]
        #     if len(v["categories"]):
        #         info["categories"] = [x['name'] for x in v['categories']]
        #     if len(v["tags"]):
        #         info["tags"] = [x['name'] for x in v['tags']]
        #     if len(v["description"]):
        #         info["description"] = v['description']
        #     self.meta_info.append(str(json.dumps(info)))

    @staticmethod
    def get_objects(uids):
        return objaverse.load_objects(uids=uids, download_processes=1)


class Objaverse_Material(object):
    def __init__(self):
        self.dir_path = config.DATA_DIR
        self.all_objaverse_materials = json.load(open(os.path.join(self.dir_path, "objaverse_random_obj_material_dict.json")))

        self.all_objects = []
        self.all_cats = []

        idx = 0

        for cat, materials in self.all_objaverse_materials.items():
            for material in materials:
                material['obj_id'] = idx
                self.all_objects.append(material)
                idx += 1

            self.all_cats.append(cat)
    
    def get_random_objs(self, num: int) -> list:
        chosen_cats = np.random.choice(self.all_cats, num)

        chosen_objects = []
        find_ambiguous = False

        for cat in chosen_cats:
            chosen_objects.extend([str(obj) for obj in self.all_objaverse_materials[cat]])
            if len(self.all_objaverse_materials[cat]) >= 2:
                find_ambiguous = True

        while not find_ambiguous:
            cat = np.random.choice(self.all_cats, 1)[0]
            if len(self.all_objaverse_materials[cat]) >= 2:
                find_ambiguous = True
                chosen_objects.extend([str(obj) for obj in self.all_objaverse_materials[cat]])

        return chosen_objects


class Objaverse_Material2(object):
    def __init__(self):
        self.dir_path = config.DATA_DIR
        self.all_objects = json.load(open(os.path.join(self.dir_path, "objaverse_random_obj_material_list_expanded.json")))

    def get_random_objs(self, num: int) -> list:
        index = random.randint(0, len(self.all_objects) - num)
        chosen_objects = self.all_objects[index:index+num]
        return chosen_objects


class ObjectFolder(object):
    def __init__(self):
        self.dir_path = config.OBJECTFOLDER_DIR
        self.obj2cate = dict()
        meta = pd.read_csv(os.path.join(self.dir_path, "objects.csv"), header=None)
        abo = pd.read_csv(os.path.join(self.dir_path, "abo_classes_3d.txt"), sep=",", header=None)
        cate_map = dict(zip(meta[0].astype(int), meta[1]))
        abo_map = dict(zip(abo[0], abo[1]))
        self.id2material = dict(zip(meta[0].astype(int), meta[3]))

        self.id2cate = dict()
        for k, v in cate_map.items():
            if v in abo_map:
                v = abo_map[v]
            self.id2cate[k] = v

        self.categories = list(set(self.id2cate.values()))
        cate2material = {x: [] for x in self.categories}
        for k, v in self.id2cate.items():
            cate2material[v].append(self.id2material[k])

        select_cate = []
        self.cate2materialset = dict()
        self.cate2materialset2 = dict()
        for k, v in cate2material.items():
            if len(set(v)) > 1:
                self.cate2materialset[k] = list(set(v))
                # print(k, set(v))
                select_cate.append(k)
            if len(v) > 1:
                self.cate2materialset2[k] = list(set(v))
        # self.cate2ids = {x: [] for x in select_cate}
        self.cate2ids = defaultdict(list)
        for k, v in self.id2cate.items():
            # if v in select_cate:
            self.cate2ids[v].append(k)

    # Before call this function, please download all ObjectFolder objects in the ObjectFolder directory

    def get_objects(self, category):
        material = re.findall("_(Iron|Wood|Plastic|Steel|Ceramic|Polycarbonate|Glass|iron|wood|plastic|steel|ceramic|polycarbonate|glass)", category)[0]

        cat = category.replace("_"+material, "").strip()
        material = material.replace("(", "").replace(")", "")
        
        ids = self.cate2ids[cat]
        if not len(ids):
            cat = cat.replace("_", " ")
            ids = self.cate2ids[cat]
        
        final_ids = []

        for id2 in ids:
            if self.id2material[id2].lower() == material.lower():
                final_ids.append(id2)

        id2 = random.choice(final_ids)
        path = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, str(id2), "model_new.obj")

        return cat, id2, material, path

    @staticmethod
    def modify_obj(fn, new_fn):
        fin = open(fn, 'r')
        fout = open(new_fn, 'w')
        lines = [line.rstrip() for line in fin]
        fin.close()

        vertices = []; normals = []; faces = []; vns = []
        header = ""
        for line in lines:
            if line.startswith('v '):
                vertice = np.float32(line.split()[1:4])
                line = "v %f %f %f"%(vertice[0], vertice[2], vertice[1])

            fout.write(line+"\n")
        fout.close()

    @staticmethod
    def normalize_pts(pts):
        out = np.array(pts, dtype=np.float32)
        center = np.mean(out, axis=0)
        out -= center
        scale = np.sqrt(np.max(np.sum(out**2, axis=1)))
        out /= scale
        return out

    @staticmethod
    def load_obj(fn):
        fin = open(fn, 'r')
        lines = [line.rstrip() for line in fin]
        fin.close()

        vertices = []; normals = []; faces = [];
        for line in lines:
            if line.startswith('v '):
                vertices.append(np.float32(line.split()[1:4]))
            elif line.startswith('f '):
                faces.append(np.int32([item.split('/')[0] for item in line.split()[1:4]]))

        return vertices, faces

    def rotate_and_normalize(self):
        for obj in tqdm(os.listdir(config.OBJECTFOLDER_OBJECTS_DIR)):
            model_file = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "model.obj")
            new_model_file = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "model_new.obj")
            print ("processing %s"%model_file)
            self.modify_obj(model_file, new_model_file)

    def generate_vertices_and_forces(self):
        for obj in tqdm(os.listdir(config.OBJECTFOLDER_OBJECTS_DIR)):
            model_file = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "model.obj")
            save_vertice_file = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "vertices.npy")
            save_force_file = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "forces.npy")
            v, f = self.load_obj(model_file)
            v = random.sample(v, 20)
            forces = np.ones((20, 3))
            v = np.vstack(v)
            np.save(save_vertice_file, v); np.save(save_force_file, forces)

    def embed_features(self):
        from msclap import CLAP
        import torch
        from subprocess import call

        clap_model = CLAP(version = '2023', use_cuda=False)

        for obj in tqdm(os.listdir(config.OBJECTFOLDER_OBJECTS_DIR)):
            try:
                audio_dir = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "results")
                feature_save_dir = os.path.join(config.OBJECTFOLDER_OBJECTS_DIR, obj, "features")
                cmd = "rm -rf %s*"%feature_save_dir
                call(cmd, shell=True)
                os.mkdir(feature_save_dir)
                if not os.path.exists(audio_dir):
                    continue
                
                audio_files = os.listdir(audio_dir)
                audio_files = [os.path.join(audio_dir, file) for file in audio_files]
                audio_embeddings = clap_model.get_audio_embeddings(audio_files)
                
                for i in range(audio_embeddings.shape[0]):
                    torch.save(audio_embeddings[i], feature_save_dir+"/"+str(i)+".pt")
            except:
                print ("failed processing clap features for %s" %obj)


if __name__ == "__main__":
    # TODO: spilt train and test set (use src_file label for audio files)
    # # hm3d = HM3D()
    # objectfolder = ObjectFolder()
    # objectfolder.prepare_adapter_data()

    objaverse = Objaverse()
    audio_set = AudioSet(training_set=True)
    audio2objaverse = json.load(open(os.path.join(config.DATA_DIR, "audio2objaverse.json"), "r"))
    audio_cate2node = {v: k for k, v in audio_set.node2name.items()}

    # obj_cate = random.choice(objaverse.categories)
    obj2audio = dict()
    for i in objaverse.categories:
        audio_cate = [k for k, v in audio2objaverse.items() if i in v]
        audio_ids = []
        nodes = []
        if len(audio_cate):
            # Check audio_cate in case GPT generates category that does not exist
            nodes = [audio_cate2node[x] for x in audio_cate if x in audio_set.node2name.values()]
            audio_ids = audio_set.get_ids(nodes)
        obj2audio[i] = audio_ids
        print(i, len(audio_ids), [audio_set.node2name[x] for x in nodes])
    json.dump(obj2audio, open(os.path.join(config.DATA_DIR, "obj2audio_ids.json"), "w"))

            # if len(nodes):
            #     node = random.choice(nodes)  # Tip: be careful about data balance problem. Categories of AudioSet is a tree.
            #     audio_ids = audio_set.get_ids([node])
            #     audio_id = random.choice(audio_ids)
            #     audio_feature, success = audio_set.get_embedding(audio_id)
            #     print(obj_cate, audio_set.node2name[node], audio_id, success, len(audio_feature) if audio_feature else None)
