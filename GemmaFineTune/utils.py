"""
utils.py
--------
Contains utility functions and global variables. It imports the necessary libraries and initializes the tokenizer and embedding model.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

from transformers import AutoTokenizer, AutoModel
from config import model_id, HFToken, CUSTOMISATION_MODEL, CUSTOMISATION_LOSS

# Hugging Face Token
import os
os.environ["HF_TOKEN"] = HFToken

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

embedding_model = None
if CUSTOMISATION_MODEL or CUSTOMISATION_LOSS:
    embedding_model = AutoModel.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
