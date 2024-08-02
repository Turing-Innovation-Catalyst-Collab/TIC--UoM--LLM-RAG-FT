"""
custom_early_stopping_callback.py
---------------------------------
Contains the CustomEarlyStoppingCallback class for early stopping during training.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

from transformers import TrainerCallback
from config import CUSTOMISATION_LOSS
import shutil
import os

class CustomEarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience=30, early_stopping_threshold=0.01):
        """
        Initialize the callback.

        Parameters:
        - early_stopping_patience (int): Number of evaluations to wait for improvement.
        - early_stopping_threshold (float): Threshold for improvement.
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.wait_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Check for improvement during evaluation.

        Parameters:
        - args (TrainingArguments): Training arguments.
        - state (TrainerState): Trainer state.
        - control (TrainerControl): Trainer control.
        - metrics (dict): Evaluation metrics.
        """
        loss_name = 'eval_custom_loss' if CUSTOMISATION_LOSS else 'eval_loss'

        current_metric = metrics.get(loss_name)

        if current_metric is not None:
            if self.best_metric is None or self._is_improvement(current_metric):
                self.best_metric = current_metric
                self.wait_count = 0
                print("Improvement found, resetting wait_count.")

                # Define the new best model directory
                model_dir = os.path.join(args.output_dir, "best_model")
                # Delete the previous best model directory
                if model_dir is not None and os.path.exists(model_dir):
                    shutil.rmtree(model_dir)

                # Save the new best model
                kwargs['model'].save_pretrained(model_dir)
                kwargs['tokenizer'].save_pretrained(model_dir)
                print(f"Best model saved at {model_dir}")
            else:
                self.wait_count += 1
                print(f"No improvement. wait_count incremented to {self.wait_count}")
                if self.wait_count >= self.early_stopping_patience:
                    control.should_training_stop = True
                    print("Early stopping triggered.")

    def _is_improvement(self, current_metric):
        """
        Check if the current metric is an improvement.

        Parameters:
        - current_metric (float): Current evaluation metric.

        Returns:
        - bool: True if the current metric is an improvement, False otherwise.
        """
        if self.best_metric is None:
            return True
        if self.early_stopping_threshold > 0:
            return current_metric < self.best_metric - self.early_stopping_threshold
        else:
            return current_metric < self.best_metric