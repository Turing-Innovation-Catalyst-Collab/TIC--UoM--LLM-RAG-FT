"""
config.py
---------
Contains configuration constants and parameters for fine-tuning the model.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0


# Model configuration
model_id = "google/gemma-2b"
QUANTISATION = True
CUSTOMISATION_MODEL = False
CUSTOMISATION_LOSS = False
DEBUG = False

HFToken = "YOUR_HF_TOKEN"

# URLs of the .xlsx files on GitHub (raw URLs)
test_url = 'https://github.com/Paladindevil/Data/blob/main/test.xlsx?raw=true'
train_url = 'https://github.com/Paladindevil/Data/blob/main/train.xlsx?raw=true'

# Fine-tuning parameters
lora_r_choices = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
lora_alpha_choices = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
lora_dropout_choices = [0.05, 0.1]
learning_rate_choices = [1e-4, 5e-4, 1e-3]
weight_decay_choices = [0.005, 0.01]
target_modules_choices = [
    ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
]

# Early stopping parameters
early_stopping_patience = 30
early_stopping_threshold = 0.01

# Training arguments
training_args = {
    "per_device_train_batch_size": 1,
    "num_train_epochs": 1000,
    "max_steps": 1000,
    "logging_steps": 1,
    "evaluation_strategy": "steps",
    "eval_steps": 1,
    "save_steps": 0,
    "save_total_limit": 0,
    "load_best_model_at_end": False,
    "disable_tqdm": True,
    "report_to": [],
}