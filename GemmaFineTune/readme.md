# Gemma Fine-Tuning

This repository contains scripts for fine-tuning the Gemma language model using custom configurations, data preparation, and evaluation metrics. The scripts include configuration settings, data preparation functions, training routines, and evaluation methods.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [License](#license)

## Overview

This project is designed to facilitate the fine-tuning and evaluation of the Gemma language model. It includes scripts for setting configurations, downloading and preparing data, training the model, and evaluating its performance using custom metrics.

## Features

- Configurable model fine-tuning parameters.
- Functions for downloading and preparing data.
- Training and evaluation scripts with early stopping.
- Custom metric computation for model evaluation.

## Installation

Ensure you have the libraries given in requirements.txt installed.

You can install these libraries using pip:

pip install -r requirements.txt

This will install all the necessary libraries listed in the `requirements.txt` file.

## Usage

### 1. Configuration

The `config.py` file contains configuration constants and parameters for fine-tuning the model. Set your model ID, URLs for data, fine-tuning parameters, early stopping parameters, and training arguments.

### 2. Data Preparation

Use `data_preparation.py` to download and prepare your data. This script includes functions to download Excel files, clean text, generate prompts, and initialises datasets.

### 3. Training

Run `main.py` to start the training process. This script will load the configurations, prepare the data, initialise the model, and perform training and evaluation.

### 4. Evaluation

`evaluation.py` includes functions for computing metrics like custom loss using cosine similarity.

## Requirements

The memory requirements will vary depending on the language model (LM) used. For the configuration provided in Gemma, you will need between 4 GB and 20 GB of graphics card memory to run the process.

## Configuration

### `config.py`

Contains configuration constants and parameters for fine-tuning the model.

- Model configuration constants:
  - `model_id = "google/gemma-2b"`
  - `QUANTISATION = True`
  - `CUSTOMISATION_MODEL = False`
  - `CUSTOMISATION_LOSS = False`
  - `DEBUG = False`
  - `HFToken = "YOUR_HF_TOKEN"`

- URLs for data:
  - `test_url = 'https://github.com/Paladindevil/Data/blob/main/test.xlsx?raw=true'`
  - `train_url = 'https://github.com/Paladindevil/Data/blob/main/train.xlsx?raw=true'`

- Fine-tuning parameters:
  - `lora_r_choices = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]`
  - `lora_alpha_choices = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`
  - `lora_dropout_choices = [0.05, 0.1]`
  - `learning_rate_choices = [1e-4, 5e-4, 1e-3]`
  - `weight_decay_choices = [0.005, 0.01]`
  - `target_modules_choices = [["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]]`

- Early stopping parameters:
  - `early_stopping_patience = 30`
  - `early_stopping_threshold = 0.01`

- Training arguments:
  - `training_args = {"per_device_train_batch_size": 1, "num_train_epochs": 1000, "max_steps": 1000, "logging_steps": 1, "evaluation_strategy": "steps", "eval_steps": 1, "save_steps": 0, "save_total_limit": 0, "load_best_model_at_end": False, "disable_tqdm": True, "report_to": []}`

## Data Preparation

### `data_preparation.py`

Contains functions for downloading and preparing data.

- `download_excel_file(url, local_path)`
- `cleanText(text)`
- `generateDualPrompt(dataPoint)`
- `initialiseData(filePath, outputJson)`
- `tokenize_function(examples, tokenizer)`
- `prepare_dataset(dataset, tokenizer)`
- `load_my_dataset(json_file)`
- `my_dataset(train_json='train.json', test_json='test.json')`
- `formatting_func(input)`

## Training

### `main.py`

Main script to run the training process. It loads the configurations, prepares the data, initialises the model, and performs the training and evaluation.

- Loads configurations from `config.py`.
- Prepares data using functions from `data_preparation.py`.
- Initialises the model and trains it using functions from `training.py`.
- Evaluates the model using metrics from `evaluation.py`.

### `training.py`

Contains functions for training the model.

- `load_model(config, model_id, bnb_config)`
- `setup_training_args(config, output_dir)`
- `train_model(model, lora_config, training_args_instance, tokenized_data)`
- `evaluate_model(trainer)`
- `train_and_evaluate(config, model_id, tokenized_data, bnb_config)`

## Evaluation

### `evaluation.py`

Contains functions for computing metrics.

- `get_sentence_embedding(sentence)`
- `compute_metrics(eval_pred)`

## Running the code

To run the code, execute `main.py`. The `config.py` file contains all the necessary configurations. You must defined your HuggingFace API Token within the given document. As highlighted in this document, refer to the [Configuration](#configuration) section to identify the appropriate files to modify.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://www.apache.org/licenses/LICENSE-2.0) file for details.