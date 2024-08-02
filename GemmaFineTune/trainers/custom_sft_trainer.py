"""
custom_sft_trainer.py
---------------------
Contains the CustomSFTTrainer class for custom fine-tuning with cosine similarity loss.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import torch
import torch.nn as nn
import torch.optim as optim
from trl import SFTTrainer
from config import DEBUG

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, tokenizer, embedding_model, *args, **kwargs):
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.optimizer = self.create_optimizer()
        self.scaler = torch.cuda.amp.GradScaler()

    def create_optimizer(self):
        """
        Create the optimizer.

        Returns:
        - optimizer (Optimizer): Created optimizer.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer

    def get_sentence_embedding(self, sentence):
        """
        Get the embedding of a sentence.

        Parameters:
        - sentence (str): Sentence to be embedded.

        Returns:
        - torch.Tensor: Sentence embedding.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")
        inputs = {k: v.to(self.embedding_model.device) for k, v in inputs.items()}
        outputs = self.embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for training.

        Parameters:
        - model (PreTrainedModel): Model to compute loss for.
        - inputs (dict): Inputs for the model.
        - return_outputs (bool): Whether to return the outputs.

        Returns:
        - loss (torch.Tensor): Computed loss.
        - outputs (Optional[ModelOutput]): Model outputs if return_outputs is True.
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']

        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        labels = labels.to(model.device)

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            if DEBUG:
                print(f"Logits: {logits}")
                print(f"Labels: {labels}")

            original_loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            print(f"Original Loss: {original_loss}")

            pred_ids = torch.argmax(logits, dim=-1)
            pred_sentences = [self.tokenizer.decode(pred_id, skip_special_tokens=True) for pred_id in pred_ids]
            label_sentences = [self.tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            pred_embeddings = torch.stack([self.get_sentence_embedding(sent) for sent in pred_sentences])
            label_embeddings = torch.stack([self.get_sentence_embedding(sent) for sent in label_sentences])

            cos_sim = nn.functional.cosine_similarity(pred_embeddings, label_embeddings, dim=-1).mean()
            loss = 1 - cos_sim
            combined_loss = loss + original_loss * 0

        combined_loss = combined_loss.to(model.device)
        return (combined_loss, outputs) if return_outputs else combined_loss

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Parameters:
        - model (PreTrainedModel): Model to train.
        - inputs (dict): Inputs for the model.

        Returns:
        - torch.Tensor: Loss value after the training step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with torch.cuda.amp.autocast():
            loss = self.compute_loss(model, inputs)
        if DEBUG:
            print(f"Loss before scaling: {loss.item()}")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()
        if DEBUG:
            print(f"Loss after scaling: {loss.item()}")
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"After backward pass: Parameter {name} gradient: {param.grad.data.norm().item()}")
                else:
                    print(f"After backward pass: Parameter {name} has no gradient.")

        return loss.detach()

    def train(self):
        """
        Train the model.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")

        self.callback_handler.on_train_begin(self.args, self.state, self.control)
        print(f"Training started with max steps: {self.args.max_steps}, gradient accumulation steps: {self.args.gradient_accumulation_steps}")

        global_step = 0
        for epoch in range(int(self.args.num_train_epochs)):
            for step, inputs in enumerate(self.get_train_dataloader()):
                loss = self.training_step(self.model, inputs)
                global_step += 1

                if (global_step) % self.args.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    print(f"Step {global_step}, Loss: {loss.item()}")

                    if global_step % self.args.eval_steps == 0:
                        eval_output = self.evaluate()
                        
                        if self.control.should_training_stop:
                            print("Early stopping triggered.")
                            break

                if global_step >= self.args.max_steps:
                    print(f"Reached max steps of {self.args.max_steps}.")
                    break

            if self.control.should_training_stop or global_step >= self.args.max_steps:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)
