import glob
import os
import json
import pickle
import random
import time
import itertools
import pandas as pd
import json

import torch.nn.functional as F

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import torch
from torch.utils.data import Dataset
import webdataset as wds
import cv2

from minigpt4.datasets.datasets.base_dataset import BaseDataset

class FeatureFaceDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):

        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.emotion_instruction_pool = [
            "Please determine which emotion label in the video represents: angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted.",

            # "Please determine which emotion label in the video represents: happy, sad, neutral, angry, worried, surprise.",
            # "Identify the displayed emotion in the video: is it happy, sad, neutral, angry, worried, or surprise?",
            # "Determine the emotional state shown in the video, choosing from happy, sad, neutral, angry, worried, or surprise.",
            # "Please ascertain the specific emotion portrayed in the video, whether it be happy, sad, neutral, angry, worried, or surprise.",
            # "Assess and label the emotion evident in the video: could it be happy, sad, neutral, angry, worried, surprise?",
        ]

        self.task_pool = [
           "emotion",
        ]

        print("ann_path: ", ann_path)
        self.ann_path = ann_path
        self.file_path = os.path.dirname(ann_path)
        with open(ann_path, 'rb') as f:
            self.tmp = pickle.load(f)
        print(('sample number:%d' % (len(self.tmp))))

        # TODO Change the logic of processing emotions
        emos = ["angry", "happy", "sad", "neutral", "frustrated", "excited", "fearful", "surprised", "disgusted"]
        self.emo2idx, self.idx2emo = {}, {}
        for ii, emo in enumerate(emos): self.emo2idx[emo] = ii
        for ii, emo in enumerate(emos): self.idx2emo[ii] = emo


    def __len__(self):
        return len(self.tmp)

    def __getitem__(self, index):
        t = self.tmp[index]
        video_name = t['id']

        video_path = os.path.join(self.vis_root, video_name + ".npy")
        if os.path.exists(video_path):
            image = self.extract_frame(video_path)
        image = Image.fromarray(image.astype('uint8'))
        image = image.convert('RGB')
        image = self.vis_processor(image)

        FaceMAE_feats, VideoMAE_feats, Audio_feats = self.get(video_name)
        if len(VideoMAE_feats.shape) == 1:
            VideoMAE_feats = VideoMAE_feats.unsqueeze(0)
        if len(Audio_feats.shape) == 1:
            Audio_feats = Audio_feats.unsqueeze(0)
        if len(FaceMAE_feats.shape) == 1:
            FaceMAE_feats = FaceMAE_feats.unsqueeze(0)
        video_features = torch.cat((FaceMAE_feats, VideoMAE_feats, Audio_feats), dim=0)

        # TODO Change the logic of processing emotions
        caption = t['emotion'] # llama2 putput only emotion class
        caption = self.text_processor(caption)
        instruction_pool = self.emotion_instruction_pool

        task = "emotion"
        emotion = self.emo2idx[t['emotion']]
        sentence = t['transcription']
        character_line = "The person in video says: {}. ".format(sentence)
        
        instruction = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(character_line, task, random.choice(instruction_pool))

        return {
            "image": image,
            "video_features": video_features,
            "instruction_input": instruction,
            "answer": caption,
            "emotion": emotion,
            "image_id": video_name
        }
    
    def extract_frame(self, video_path):
        video_capture = np.load(video_path)
        video_capture = video_capture[0]
        return video_capture

    def get(self, video_name):
        # FaceMAE feature
        FaceMAE_feats_path = os.path.join(self.file_path, 'mae_features', video_name + '.npy')
        FaceMAE_feats = torch.tensor(np.load(FaceMAE_feats_path))

        # VideoMAE feature
        VideoMAE_feats_path = os.path.join(self.file_path, 'maeV_features', video_name + '.npy')
        VideoMAE_feats = torch.tensor(np.load(VideoMAE_feats_path))

        # Audio feature
        Audio_feats_path = os.path.join(self.file_path, 'hubert_features', video_name + '.npy')
        Audio_feats = torch.tensor(np.load(Audio_feats_path))

        return FaceMAE_feats, VideoMAE_feats, Audio_feats
