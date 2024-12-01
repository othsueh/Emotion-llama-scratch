import yaml
import os
from sklearn.model_selection import KFold
import pickle  # Assuming your dataset is stored as a pickle file.

# Paths to your YAML files and dataset
train_yaml_path = "path/to/train_config.yaml"
eval_yaml_path = "path/to/eval_config.yaml"
dataset_path = "/datas/store163/othsueh/Corpus/IEMOCAP/data_collected_woother.pickle"

# Load dataset
with open(dataset_path, "rb") as f:
    data = pickle.load(f)  # Adjust based on your data structure.

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in kf.split(data):
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]

    # Save fold-specific train and test data
    train_data_path = f"/path/to/save/train_fold_{fold}.pickle"
    test_data_path = f"/path/to/save/test_fold_{fold}.pickle"

    with open(train_data_path, "wb") as f:
        pickle.dump(train_data, f)
    with open(test_data_path, "wb") as f:
        pickle.dump(test_data, f)

    # Update train YAML
    with open(train_yaml_path, "r") as file:
        train_yaml = yaml.safe_load(file)
    train_yaml["ann_path"] = train_data_path
    train_yaml["ckpt"] = f"/path/to/save/ckpt_fold_{fold}.pth"  # Optional

    with open(train_yaml_path, "w") as file:
        yaml.dump(train_yaml, file)

    # Update eval YAML
    with open(eval_yaml_path, "r") as file:
        eval_yaml = yaml.safe_load(file)
    eval_yaml["eval_file_path"] = test_data_path
    eval_yaml["ckpt"] = f"/path/to/save/ckpt_fold_{fold}.pth"  # Ensure consistency with train config

    with open(eval_yaml_path, "w") as file:
        yaml.dump(eval_yaml, file)

    # Train and evaluate the model for the current fold
    os.system(f"python train_model.py --config {train_yaml_path}")
    os.system(f"python eval_model.py --config {eval_yaml_path}")

    fold += 1
