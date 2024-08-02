"""
main.py
-------
Main script to run the training process. It loads the configurations, prepares the data, 
initializes the model, and performs the training and evaluation.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import os
import warnings
import torch
from itertools import product
from data_preparation import download_excel_file, initialiseData, prepare_dataset, my_dataset
from config import (
    model_id, QUANTISATION, HFToken, lora_r_choices, lora_alpha_choices, 
    lora_dropout_choices, learning_rate_choices, weight_decay_choices,
    target_modules_choices, test_url, train_url
)
from utils import tokenizer
from training import train_and_evaluate

warnings.filterwarnings("ignore")

# Hugging Face Token
os.environ["HF_TOKEN"] = HFToken

# Local paths where the files will be saved
test_local_path = 'test.xlsx'
train_local_path = 'train.xlsx'

# Download the files if they don't exist
download_excel_file(test_url, test_local_path)
download_excel_file(train_url, train_local_path)

# Initialize data for both test and train files
initialiseData('test.xlsx', 'test.json')
initialiseData('train.xlsx', 'train.json')

# Load and tokenize the datasets
data = my_dataset()
tokenized_data = prepare_dataset(data, tokenizer)

bnb_config = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': torch.bfloat16
} if QUANTISATION else {}

best_loss = float("inf")
best_config = None
best_model_dir = "best_4bit" if QUANTISATION else "best_32bit"

# Iterate over all combinations of hyperparameters
configs = list(product(
    lora_r_choices, lora_alpha_choices, lora_dropout_choices,
    learning_rate_choices, weight_decay_choices, target_modules_choices))

skipped_configs = []

recovery = True
i = 0
while i < len(configs):
    lora_r, lora_alpha, lora_dropout, learning_rate, weight_decay, target_modules = configs[i]

    config = {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "target_modules": target_modules,
    }

    output_dir = f"outputs_4bit/config_{lora_r}_{lora_alpha}_{lora_dropout}_{learning_rate}_{weight_decay}" if QUANTISATION else f"outputs_32bit/config_{lora_r}_{lora_alpha}_{lora_dropout}_{learning_rate}_{weight_decay}"

    # Check if the directory exists
    if os.path.exists(output_dir):
        print(f"Skipping existing config: {config}")
        skipped_configs.append(config)
        i += 1
        continue

    # If we have skipped configurations, go back and handle the last skipped one
    if recovery and skipped_configs:
        print(f"Re-running skipped config: {skipped_configs[-1]}")
        config = skipped_configs.pop()
        recovery = False
        i -= 1

    print(f"Training with config: {config}")
    loss, trainer = train_and_evaluate(config, model_id, tokenized_data, bnb_config)
    print(f"Loss: {loss}")

    if loss < best_loss:
        best_loss = loss
        best_config = config

    i += 1
    
print(f"Best config: {best_config}")
print(f"Best loss: {best_loss}")