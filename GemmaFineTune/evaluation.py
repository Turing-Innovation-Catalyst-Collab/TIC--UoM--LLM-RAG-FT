"""
evaluation.py
-------------
Contains functions for computing metrics.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from utils import tokenizer, embedding_model

def get_sentence_embedding(sentence):
    """
    Get the embedding of a sentence.

    Parameters:
    - sentence (str): Sentence to be embedded.

    Returns:
    - numpy.ndarray: Sentence embedding.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(embedding_model.device) for k, v in inputs.items()}
    outputs = embedding_model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding.cpu().detach().numpy()

def compute_metrics(eval_pred):
    """
    Compute the custom loss metric.

    Parameters:
    - eval_pred (tuple): Tuple containing logits and labels.

    Returns:
    - dict: Dictionary containing the custom loss.
    """
    logits, labels = eval_pred

    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    pred_ids = torch.argmax(logits, dim=-1)
    pred_sentences = [tokenizer.decode(pred_id, skip_special_tokens=True) for pred_id in pred_ids]
    label_sentences = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

    pred_embeddings = np.array([get_sentence_embedding(sent) for sent in pred_sentences])
    label_embeddings = np.array([get_sentence_embedding(sent) for sent in label_sentences])

    if pred_embeddings.ndim == 3:
        pred_embeddings = pred_embeddings.squeeze(1)
    if label_embeddings.ndim == 3:
        label_embeddings = label_embeddings.squeeze(1)

    cos_sim = cosine_similarity(pred_embeddings, label_embeddings).mean()
    return {"custom_loss": 1 - cos_sim}