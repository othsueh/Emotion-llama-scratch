import yaml
import os
import pickle  # Assuming your dataset is stored as a pickle file.

# Paths to your YAML files and dataset
train_yaml_path = "/datas/store163/othsueh/Emotion-LLaMA/minigpt4/configs/datasets/firstface/featureface.yaml"
config_yaml_path = "/datas/store163/othsueh/Emotion-LLaMA/train_configs/Emotion-LLaMA_finetune.yaml"
eval_yaml_path = "/datas/store163/othsueh/Emotion-LLaMA/eval_configs/eval_emotion.yaml"
test_path = "/datas/store163/othsueh/Corpus/IEMOCAP/data_collected_woother_test.pickle"

epoch = 29 # 30 epochs
folds = [1, 2, 3, 4, 5]  # Number of folds
for fold in folds: 
    train_data_path = f"/datas/store163/othsueh/Corpus/IEMOCAP/Cross_validation/fold{fold}_train.pickle"
    test_data_path = f"/datas/store163/othsueh/Corpus/IEMOCAP/Cross_validation/fold{fold}_val.pickle"
    ckpt_path = f"/datas/store163/othsueh/Emotion-LLaMA/checkpoints/save_checkpoint/ckpt_fold_{fold}/"
    # Update train YAML
    with open(train_yaml_path, "r") as file:
        train_yaml = yaml.safe_load(file)

    train_yaml["datasets"]["feature_face_caption"]["build_info"]["ann_path"] = train_data_path

    with open(train_yaml_path, "w") as file:
        yaml.dump(train_yaml, file)

    # Update config YAML
    with open(config_yaml_path, "r") as file:
        config_yaml = yaml.safe_load(file)

    os.makedirs(ckpt_path, exist_ok=True)

    config_yaml["run"]["output_dir"] = ckpt_path

    if fold == 1:
        config_yaml["model"]["ckpt"] = "/datas/store163/othsueh/Emotion-LLaMA/checkpoints/save_checkpoint/MERR_train_checkpoint.pth"
    else:
        last_ckpt_path = f"/datas/store163/othsueh/Emotion-LLaMA/checkpoints/save_checkpoint/ckpt_fold_{fold-1}/"
        config_yaml["model"]["ckpt"] = last_ckpt_path + f"checkpoint_{epoch}.pth"  

    with open(config_yaml_path, "w") as file:
        yaml.dump(config_yaml, file) 

    # Update eval YAML
    with open(eval_yaml_path, "r") as file:
        eval_yaml = yaml.safe_load(file)

    eval_yaml["evaluation_datasets"]["feature_face_caption"]["eval_file_path"] = test_data_path
    eval_yaml["model"]["ckpt"] = ckpt_path + f"checkpoint_{epoch}.pth"  # Ensure consistency with train config
    eval_yaml["run"]["cross_turn"] = fold

    with open(eval_yaml_path, "w") as file:
        yaml.dump(eval_yaml, file)

    # Train and evaluate the model for the current fold
    try: 
        # Train the model
        os.system(f"torchrun --nproc-per-node 1 train.py --cfg-path {config_yaml_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    
    try: 
        # Validate the model
        os.system(f"torchrun --nproc_per_node 1 eval_emotion.py --cfg-path {eval_yaml_path} --dataset feature_face_caption")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

# Test the model
with open(eval_yaml_path, "r") as file:
    eval_yaml = yaml.safe_load(file)

eval_yaml["evaluation_datasets"]["feature_face_caption"]["eval_file_path"] = test_path
eval_yaml["run"]["cross_turn"] = "test"

with open(eval_yaml_path, "w") as file:
    yaml.dump(eval_yaml, file)

try: 
    # Validate the model
    os.system(f"torchrun --nproc_per_node 1 eval_emotion.py --cfg-path {eval_yaml_path} --dataset feature_face_caption")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
