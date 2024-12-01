import csv
import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.registry import registry

from minigpt4.datasets.datasets.iemocap import FeatureFaceDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from minigpt4.datasets.data_utils import prepare_sample


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='feature_face_caption', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

print("cfg:", args)
cfg = Config(args)


model, vis_processor = init_model(args)

model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

save_path = cfg.run_cfg.save_path

text_processor_cfg = cfg.datasets_cfg.feature_face_caption.text_processor.train
text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
vis_processor_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)


print(args.dataset)
if 'feature_face_caption' in args.dataset:
    eval_file_path = cfg.evaluation_datasets_cfg["feature_face_caption"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["feature_face_caption"]["image_path"]
    batch_size = cfg.evaluation_datasets_cfg["feature_face_caption"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["feature_face_caption"]["max_new_tokens"]
    print(eval_file_path)
    print(img_path)
    print(batch_size)
    print(max_new_tokens)

    data = FeatureFaceDataset(vis_processor, text_processor, img_path, eval_file_path)
    # print(data)
    # print(data[0])
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    targets_list = []  
    answers_list = [] 
    names_list   = []
    for batch in eval_dataloader:
        images = batch['image']
        instruction_input = batch['instruction_input']
        # print("instruction_input:", instruction_input)
        targets = batch['answer']
        video_features = batch['video_features']

        texts = prepare_texts(instruction_input, conv_temp)
        answers = model.generate(images, video_features, texts, max_new_tokens=max_new_tokens, do_sample=False)
        
        for j in range(len(answers)):
            # print("raw answer:", answers[j])
            answers[j] = answers[j].split(" ")[-1]
            targets[j] = targets[j].split(" ")[-1]
            if answers[j] not in ["angry", "happy", "sad", "neutral", "frustrated", "excited", "fearful", "surprised", "disgusted","other"]:
                print("Error: ", answers[j], " Target:", targets[j])
                answers[j] = 'neutral'
        
        targets_list.extend(targets)  
        answers_list.extend(answers)
        names_list.extend(batch['image_id'])

    accuracy = accuracy_score(targets_list, answers_list)
    precision = precision_score(targets_list, answers_list, average='weighted')
    recall = recall_score(targets_list, answers_list, average='weighted')
    f1 = f1_score(targets_list, answers_list, average='weighted')
    
    # Save results to a file
    with open(os.path.join(save_path, 'eval_results.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("Targets: " + ', '.join(targets_list) + "\n")
        f.write("Answers: " + ', '.join(answers_list) + "\n")
    # Plot confusion matrix
    emotion_list = ["angry", "happy", "sad", "neutral", "frustrated", "excited", "fearful", "surprised", "disgusted", "other"]
    cm = confusion_matrix(targets_list, answers_list, labels=emotion_list)
    plt.figure(figsize=(10, 8))
    sn.heatmap(cm, annot=True, cmap="mako_r", xticklabels=emotion_list, yticklabels=emotion_list)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
    plt.close()

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print(cm)
    
# torchrun  --nproc_per_node 1 eval_emotion.py --cfg-path eval_configs/eval_emotion.yaml --dataset feature_face_caption
