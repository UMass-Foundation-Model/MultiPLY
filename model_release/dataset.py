from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
import os
import orjson
import torch
import random
from itertools import chain
from easydict import EasyDict
import json
import numpy as np
from PIL import Image

SCENE_TOKEN = "<scene>"
VISUAL_TOKEN = "<visual>"
TEMP_TOKEN = "<temperature>"
TACTILE_TOKEN = "<tactile>"
SOUND_TOKEN = "<sound>"
AMBIENT_TOKEN = "<ambient>"
GET_VISUAL_TOKEN = "<observe>"
GET_TACTILE_TOKEN = "<touch>"
GET_SOUND_TOKEN = "<hit>"
SELECT_TOKEN = "<select>"
NAV_TOKEN = "<nav>"
PICK_TOKEN = "<pick-up>"
PICK_DOWN_TOKEN = "<pick-down>"
EXPLORE_TOKEN = "<look-around>"

class MultisensoryDataset(Dataset):
    def __init__(
            self, json_path,
            tokenizer, max_length: int,
            scene_token=SCENE_TOKEN,
            visual_token=VISUAL_TOKEN,
            tactile_token=TACTILE_TOKEN,
            sound_token=SOUND_TOKEN,
            get_visual_token=GET_VISUAL_TOKEN,
            get_tactile_token=GET_TACTILE_TOKEN,
            get_sound_token=GET_SOUND_TOKEN,

    ):
        assert os.path.exists(json_path)
        self.items = orjson.loads(open(json_path).read())

        self.tokenizer = tokenizer

        self.scene_token = scene_token
        self.visual_token = visual_token
        self.tactile_token = tactile_token
        self.sound_token = sound_token
        self.get_visual_token = get_visual_token
        self.get_tactile_token = get_tactile_token
        self.get_sound_token = get_sound_token

        self.scene_token_id = self.tokenizer(self.scene_token).input_ids[-1]
        self.visual_token_id = self.tokenizer(self.visual_token).input_ids[-1]
        self.tactile_token_id = self.tokenizer(self.tactile_token).input_ids[-1]
        self.sound_token_id = self.tokenizer(self.sound_token).input_ids[-1]
        self.get_sound_token_id = self.tokenizer(self.get_sound_token).input_ids[-1]
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def _get_text_dict(self, item):
        return dict(
            question="Is the bed soft or hard?",
            answer="soft",
        )

    def _get_scene_feature(self, item):
        if "scene" in item: 
            features = []
            folder = item["scene"]
            bboxes = json.load(open(os.path.join("./dataset/bboxes", folder+".json")))
            path = os.path.join("./dataset/feature_dict", folder)

            k = 0
            for bbox in bboxes:
                if "id" in bbox:
                    if not str(bbox["id"]) + ".pt" in os.listdir(path): continue
                    feature = torch.load(os.path.join(path, str(bbox['id']) + ".pt"), map_location=torch.device('cpu')).unsqueeze(0)
                else:                    
                    feature = torch.load(os.path.join(path, str(10000+k) + ".pt"), map_location=torch.device('cpu')).unsqueeze(0)
                    k += 1

                features.append(feature)

            features = torch.cat(features)

            return features
        else:
            return torch.randn(256, 1024)

    def _get_visual_feature(self, item):
        if "visual" in item: 
            visual = 10000 + int(item["visual"])
            folder = item["scene"]
            path = os.path.join("./datasetg/feature_dict", folder)
            feature = torch.load(os.path.join(path, str(visual) + ".pt"), map_location=torch.device('cpu')).unsqueeze(0)

            return feature
        else:
            return torch.randn(256, 1024)

    def _get_tactile_feature(self, item):
        if "tactile_reading" in item:
            tactile_reading = torch.load(os.path.join("./dataset/data5", item["tactile_reading"], "marker4.pt"), map_location=torch.device('cpu'))
            tactile_reading = tactile_reading.mean(1)

            return tactile_reading

    def _get_temperature_feature(self, item):
        if "temperature" in item:
            if item["temperature"] in item:
                temperature = torch.load(os.path.join("./dataset/data4", item["temperature_reading"], "temp.png"), map_location=torch.device('cpu'))

            return temperature
        return torch.randn(random.randint(1, 4), 1024)

    def _get_sound_feature(self, item):
        if "impact_sound" in item: 
            impact_sound = torch.load(os.path.join("./dataset", "impact_sound_" + str(item["impact_sound"]) + "_0", "impact_sound", "0.pt")).unsqueeze(0)
            return impact_sound
        elif "scene_id" in item:
            sound = torch.load(os.path.join("./dataset/audioset/embedding", item["scene_id"]+".pt"))
            return sound
        else:
            return torch.randn(random.randint(1, 4), 1024)

    def collate_wrapper(self, batch):
        max_length = max(b.length for b in batch)
        max_scene_length = max(b.scene_feature.shape[0] for b in batch)
        
        scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
        prediction = torch.zeros((len(batch), max_scene_length))

        for (j,b) in enumerate(batch):
            scene_feature[j, :b.scene_feature.shape[0]] = b.scene_feature
            prediction[j, :b.scene_feature.shape[0]] = b.prediction


        return EasyDict(
            input_ids=torch.cat([b.input_ids for b in batch])[...,:max_length],
            attention_mask=torch.cat([b.attention_mask for b in batch])[...,:max_length],
            scene_feature=scene_feature,
            visual_feature=torch.cat([b.visual_feature for b in batch]),
            tactile_feature=torch.cat([b.tactile_feature for b in batch]),
            temperature_feature=torch.cat([b.temperature_feature for b in batch]),
            sound_feature=torch.cat([b.sound_feature for b in batch]),
            scene_insert_loc=list(chain.from_iterable([[[batch_idx, x] for x in b.scene_insert_loc] for batch_idx, b in enumerate(batch)])),
            visual_insert_loc=list(chain.from_iterable([[[batch_idx, x] for x in b.visual_insert_loc] for batch_idx, b in enumerate(batch)])),
            tactile_insert_loc=list(chain.from_iterable([[[batch_idx, x] for x in b.tactile_insert_loc] for batch_idx, b in enumerate(batch)])),
            sound_insert_loc=list(chain.from_iterable([[[batch_idx, x] for x in b.sound_insert_loc] for batch_idx, b in enumerate(batch)])),
            prediction = prediction,
            max_scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch])
        )

    def __getitem__(self, idx):
        try:
            current_item = self.items[idx]
            scene_feature = self._get_scene_feature(current_item)
            text_dict = self._get_text_dict(current_item)
            visual_feature = self._get_visual_feature(current_item)
            tactile_feature = self._get_tactile_feature(current_item)
            
            sound_feature = self._get_sound_feature(current_item)
            
            text = f'Question: {current_item["question"]} Answer: {current_item["answer"]} {self.tokenizer.eos_token}'.replace(self.tactile_token, self.tactile_token*len(tactile_feature)).replace(self.scene_token, self.scene_token*len(scene_feature)).replace(self.sound_token, self.sound_token*len(sound_feature)).replace(self.visual_token, self.visual_token*len(visual_feature))
            assert self.max_length > len(scene_feature) # make sure that scene feature is never truncated
            text = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding='max_length')
            
            input_ids = text["input_ids"]
            length = torch.nonzero(input_ids).shape[0]
            
            attention_mask = text["attention_mask"]
            scene_insert_loc = (input_ids == self.scene_token_id).nonzero()[:1, 1].reshape(-1).tolist()
            visual_insert_loc = (input_ids == self.visual_token_id).nonzero()[:, 1].reshape(-1).tolist()
            tactile_insert_loc = (input_ids == self.tactile_token_id).nonzero()[:, 1].reshape(-1).tolist()
            temperature_insert_loc = (input_ids == self.temperature_token_id).nonzero()[:, 1].reshape(-1).tolist()
            sound_insert_loc = (input_ids == self.sound_token_id).nonzero()[:, 1].reshape(-1).tolist()
            
            visual_feature = visual_feature[:len(visual_insert_loc)]
            tactile_feature = tactile_feature[:len(tactile_insert_loc)]
            temperature_feature = temperature_feature[:len(temperature_insert_loc)]
            sound_feature = sound_feature[:len(sound_insert_loc)]

            if "prediction" in current_item:
                prediction = current_item['prediction']
            else:
                prediction = [-1 for tok in range(len(scene_feature))]

            prediction = torch.tensor(current_item['prediction'])
            prediction[prediction>0] = 1
            prediction = prediction.float()

            return EasyDict(
                text=text,
                input_ids=input_ids,
                length=length,
                attention_mask=attention_mask,
                scene_feature=scene_feature,
                visual_feature=visual_feature,
                tactile_feature=tactile_feature,
                temperature_feature=temperature_feature,
                sound_feature=sound_feature,
                scene_insert_loc=scene_insert_loc,
                visual_insert_loc=visual_insert_loc,
                tactile_insert_loc=tactile_insert_loc,
                sound_insert_loc=sound_insert_loc,
                prediction = prediction
            )
        except:
            # print ("cannot find feature %d"%idx)
            return self.__getitem__(idx-1)
