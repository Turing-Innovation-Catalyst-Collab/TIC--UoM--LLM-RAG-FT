"""
training.py
-----------
Contains functions for training the model.

This script is used to train the model with the given configuration. It initializes the model, sets up the training arguments, trains the model, and evaluates it.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import gc
import os
import shutil
import re
import torch
from transformers import AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from trainers import CustomSFTTrainer, DebugSFTTrainer, CustomEarlyStoppingCallback
from peft import LoraConfig
from evaluation import compute_metrics
from config import CUSTOMISATION_MODEL, CUSTOMISATION_LOSS, QUANTISATION, DEBUG, early_stopping_patience, early_stopping_threshold, training_args
from utils import tokenizer, embedding_model
from data_preparation import formatting_func

def load_model(config, model_id, bnb_config):
    """
    Initialize the model with the given configuration.
    """
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )

    model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_id,
        **bnb_config,
        device_map={"": 0},
        token=os.environ['HF_TOKEN']
    )

    return model, lora_config

def setup_training_args(config, output_dir):
    """
    Setup training arguments with the given configuration.
    """
    common_training_args = training_args.copy()
    common_training_args.update({
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "output_dir": output_dir,
        "metric_for_best_model": "eval_custom_loss" if CUSTOMISATION_LOSS else "eval_loss",
        "optim": "paged_adamw_8bit" if QUANTISATION else "paged_adamw_32bit",
        "fp16": True,
    })

    return TrainingArguments(**common_training_args)

def train_model(model, lora_config, training_args_instance, tokenized_data):
    """
    Train the model with the given configuration.
    """
    early_stopping_callback = CustomEarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold
    )

    trainer_args = {
        "model": model,
        "train_dataset": tokenized_data["train"],
        "eval_dataset": tokenized_data["test"],
        "args": training_args_instance,
        "formatting_func": formatting_func,
        "compute_metrics": compute_metrics if CUSTOMISATION_LOSS else None,
        "peft_config": lora_config,
        "callbacks": [early_stopping_callback],
    }

    if CUSTOMISATION_MODEL:
        trainer_args["tokenizer"] = tokenizer
        trainer_args["embedding_model"] = embedding_model

    trainer_class = CustomSFTTrainer if CUSTOMISATION_MODEL else DebugSFTTrainer if DEBUG else SFTTrainer
    trainer = trainer_class(**trainer_args)

    trainer.train()
    return trainer

def evaluate_model(trainer):
    """
    Evaluate the trained model and return the loss.
    """
    eval_result = trainer.evaluate()
    return eval_result["eval_custom_loss"] if CUSTOMISATION_LOSS else eval_result["eval_loss"]

def train_and_evaluate(config, model_id, tokenized_data, bnb_config):
    """
    Function to initialize and train the model with the given configuration.

    Parameters:
    - config (dict): Configuration dictionary containing hyperparameters.
    - model_id (str): The model identifier.
    - tokenized_data (dict): The tokenized dataset.
    - bnb_config (dict): Configuration for bit and byte settings.

    Returns:
    - loss (float): The evaluation loss.
    - trainer (Trainer): The trained model.
    """
    model, lora_config = load_model(config, model_id, bnb_config)
    
    output_dir = f"outputs_4bit/config_{config['lora_r']}_{config['lora_alpha']}_{config['lora_dropout']}_{config['learning_rate']}_{config['weight_decay']}" if QUANTISATION else f"outputs_32bit/config_{config['lora_r']}_{config['lora_alpha']}_{config['lora_dropout']}_{config['learning_rate']}_{config['weight_decay']}"
    
    training_args_instance = setup_training_args(config, output_dir)
    
    trainer = train_model(model, lora_config, training_args_instance, tokenized_data)
    loss = evaluate_model(trainer)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return loss, trainer