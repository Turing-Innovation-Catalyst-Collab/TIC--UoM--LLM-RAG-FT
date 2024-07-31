"""
debug_sft_trainer.py
--------------------
Contains the DebugSFTTrainer class for debugging purposes.
"""

# SPDX-FileCopyrightText: [2024] University of Manchester

# SPDX-License-Identifier: apache-2.0

import torch
import torch.optim as optim
from trl import SFTTrainer

class DebugSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(DebugSFTTrainer, self).__init__(*args, **kwargs)
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

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss for training with debugging information.

        Parameters:
        - model (PreTrainedModel): Model to compute loss for.
        - inputs (dict): Inputs for the model.
        - return_outputs (bool): Whether to return the outputs.

        Returns:
        - loss (torch.Tensor): Computed loss.
        - outputs (Optional[ModelOutput]): Model outputs if return_outputs is True.
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        print(f"Logits: {outputs.logits}")
        print(f"Labels: {labels}")
        print(f"Loss: {loss}")

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs with debugging information.

        Parameters:
        - model (PreTrainedModel): Model to train.
        - inputs (dict): Inputs for the model.

        Returns:
        - torch.Tensor: Loss value after the training step.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        print(f"Loss before scaling: {loss.item()}")

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.scaler.scale(loss).backward()

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