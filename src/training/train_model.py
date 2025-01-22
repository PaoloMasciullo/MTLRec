import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.utils import kl_divergence_gaussian
from src.utils.mtl import PCGrad
from src.utils.torch_utils import get_device
from src.evaluation.utils import group_by_id
import src.evaluation.metrics as metrics
import src.layers.losses as losses


class Trainer:
    def __init__(self,
                 model,
                 evaluator,
                 monitor_metric="AucScore()",
                 monitor_mode="max",
                 optimizer="Adam",
                 optimizer_params=None,
                 loss_function="BCELoss()",
                 task="binary-classification",
                 device=-1,
                 expid="",
                 log_dir='./logs/',
                 save_path='./checkpoints/'):
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.epochs_no_improve = None
        self.patience = None
        self.expid = expid
        self.save_path = save_path + self.expid
        self.model = model
        self.optimizer = getattr(optim, optimizer)(model.parameters(), **optimizer_params)
        if loss_function.startswith("BPRLoss") or loss_function.startswith("FocalLoss"):
            self.loss_function = eval(f"losses.{loss_function}")
        else:
            self.loss_function = eval(f"nn.{loss_function}")
        self.monitor_metric = eval(f"metrics.{monitor_metric}")
        self.monitor_mode = monitor_mode
        self.task = task
        self.device = get_device(device)
        self.model.to(device=self.device)
        self.writer = SummaryWriter(log_dir + self.expid + "/" + datetime.datetime.now().strftime("%B-%d-%Y_%I-%M%p"))
        self.evaluator = evaluator
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        if self.monitor_mode == 'min':
            self.best_val_metric = float('inf')
        elif self.monitor_mode == 'max':
            self.best_val_metric = -float('inf')
        else:
            raise ValueError(f"Monitor mode value '{self.monitor_mode}' not supported")

    def fit(self, train_loader, val_loader, epochs, patience, max_gradient_norm=10.):
        self.epochs_no_improve = 0  # Track how many epochs without improvement
        self.patience = patience
        early_stop = False

        for epoch in range(epochs):
            print(f"\nEPOCH {epoch + 1}/{epochs}:")

            # Training phase with tqdm progress bar
            self._train_one_epoch(train_loader, epoch, max_gradient_norm)
            # Validation phase with tqdm progress bar
            val_metric = self._evaluate(val_loader, epoch)

            early_stop = self._early_stop(val_metric)
            if early_stop:
                break

        if not early_stop:
            print("Training finished without early stopping.")

        # Load the best model checkpoint after training
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'best_model.pth'), weights_only=True))

    def _early_stop(self, val_metric):
        # Early stopping check based on monitor mode
        if (self.monitor_mode == 'min' and val_metric < self.best_val_metric) or (
                self.monitor_mode == 'max' and val_metric > self.best_val_metric):
            print(f"   Performance on validation improved. Saving model...")
            self.best_val_metric = val_metric
            self.epochs_no_improve = 0  # Reset early stopping counter
            # Save the best model checkpoint
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "best_model.pth"))
        else:
            self.epochs_no_improve += 1
            print(
                f"   No improvement on validation. Early stop patience: {self.epochs_no_improve}/{self.patience}")

        # Check for early stopping condition
        if self.epochs_no_improve >= self.patience:
            print(f"Early stopping triggered. Stopping training.")
            return True
        else:
            return False

    def _train_one_epoch(self, train_loader, epoch, max_gradient_norm):
        self.model.train()
        total_loss = 0
        total_steps = 0

        # Progress bar with tqdm
        progress_bar = tqdm(train_loader, desc=f"   Training Epoch {epoch + 1}", file=sys.stdout, delay=0.1)

        for features, labels in progress_bar:
            self.optimizer.zero_grad()
            labels = self._get_labels(labels)
            features = self._get_features(features)

            loss = self._train_step(features, labels, max_gradient_norm)

            total_loss += loss.item()  # Sum the loss over the batch
            total_steps += 1

            # Update tqdm progress bar
            progress_bar.set_postfix(loss=total_loss / total_steps)

        avg_loss = total_loss / total_steps

        # Log training metrics to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch + 1)
        print(f"   Train Loss: {avg_loss:.4f}")

    def _train_step(self, features, labels, max_gradient_norm):
        outputs = self.model(features)
        loss = self._compute_loss(outputs, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)
        self.optimizer.step()
        return loss

    def _compute_loss(self, y_pred, y_true):
        return self.loss_function(y_pred, y_true)

    def _get_labels(self, labels):
        if self.task == "multiclass-classification":
            return labels.to(self.device).long()
        elif self.task in ["binary-classification", "regression"]:
            return labels.view(-1, 1).to(self.device).float()
        elif self.task == "ranking":
            return labels.to(self.device).float()
        else:
            raise ValueError(f"Task {self.task} not supported.")

    def _get_features(self, features):
        return {feature: tensor.to(self.device) for feature, tensor in features.items()}

    def _evaluate(self, val_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_steps = 0
            outputs_list, labels_list, group_id_list = [], [], []
            # Progress bar with tqdm
            progress_bar = tqdm(val_loader, desc=f"   Validating Epoch {epoch + 1}", file=sys.stdout, delay=0.1)
            for features, labels in progress_bar:
                labels = self._get_labels(labels)
                features = self._get_features(features)

                outputs = self.model(features)
                loss = self._compute_loss(outputs, labels)

                total_loss += loss.item()  # Sum the loss over the batch
                total_steps += 1

                if self.task == "multiclass-classification":
                    outputs_list.extend(outputs.data.cpu().numpy())
                    labels_list.extend(labels.data.cpu().numpy().reshape(-1))
                elif self.task in ["binary-classification", "regression"]:
                    outputs_list.extend(outputs.data.cpu().numpy().reshape(-1))
                    labels_list.extend(labels.data.cpu().numpy().reshape(-1))
                elif self.task == "ranking":
                    outputs_list.extend(outputs.data.cpu().numpy())
                    labels_list.extend(labels.data.cpu().numpy())
                else:
                    raise ValueError(f"Task {self.task} not supported.")

                if self.model.feature_map.get("group_id", None):
                    group_id_list.extend(features[self.model.feature_map["group_id"]].cpu().numpy().reshape(-1))

                # Update tqdm progress bar
                progress_bar.set_postfix(loss=total_loss / total_steps)

            avg_loss = total_loss / total_steps

            # Convert to numpy arrays
            outputs = np.array(outputs_list, np.float64)
            labels = np.array(labels_list, np.float64)

            if self.monitor_metric.type == "group_metric" and self.task != "ranking":
                if self.model.feature_map.get("group_id", None):
                    # Group outputs and labels
                    group_ids = np.array(group_id_list, np.int32)
                    labels, outputs = group_by_id(group_ids, outputs, labels)
                else:
                    raise ValueError(
                        f"The monitor metric {self.monitor_metric.name} is a group_metric but no group_id is available in the data")

            # Compute the metric using grouped data
            metric = self.monitor_metric(labels, outputs)

            # Log validation metrics to TensorBoard
            self.writer.add_scalar('Loss/val', avg_loss, epoch + 1)
            self.writer.add_scalar(f'{self.monitor_metric.name}/val', metric, epoch + 1)

            print(f"   Val Loss: {avg_loss:.4f}")
            print(f"   Val {self.monitor_metric.name}: {metric:.4f}")
            return metric

    def evaluate_test(self, data_loader):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            outputs_list = []
            labels_list = []
            group_id_list = []
            # Progress bar with tqdm
            progress_bar = tqdm(data_loader, desc=f"   Evaluating Model on Test", file=sys.stdout, delay=0.1)
            for batch in progress_bar:
                features, labels = batch
                labels = self._get_labels(labels)
                features = self._get_features(features)

                outputs = self.model(features)

                if self.task == "multiclass-classification":
                    outputs_list.extend(outputs.data.cpu().numpy())
                    labels_list.extend(labels.data.cpu().numpy().reshape(-1))
                elif self.task in ["binary-classification", "regression"]:
                    outputs_list.extend(outputs.data.cpu().numpy().reshape(-1))
                    labels_list.extend(labels.data.cpu().numpy().reshape(-1))
                elif self.task == "ranking":
                    outputs_list.extend(outputs.data.cpu().numpy())
                    labels_list.extend(labels.data.cpu().numpy())
                else:
                    raise ValueError(f"Task {self.task} not supported.")

                if self.model.feature_map.get("group_id", None):
                    group_id_list.extend(features[self.model.feature_map["group_id"]].cpu().numpy().reshape(-1))

            # Convert to numpy arrays
            outputs = np.array(outputs_list, np.float64)
            labels = np.array(labels_list, np.float64)
            if self.model.feature_map.get("group_id", None):
                group_ids = np.array(group_id_list, np.int32)

            evaluation_results = self.evaluator.evaluate(labels=labels, predictions=outputs, group_ids=group_ids,
                                                         task=self.task)
            print("Evaluation Results on Test Set:")
            for metric, value in evaluation_results.items():
                if metric == "Confusion Matrix":
                    print(f"{metric}: \n{value}")
                else:
                    print(f"{metric}: {value:.4f}")


class MultitaskTrainer(Trainer):
    def __init__(self,
                 model,
                 evaluator,
                 monitor_metric=None,
                 monitor_metric_weight=None,
                 monitor_mode="max",
                 optimizer="Adam",
                 optimizer_params=None,
                 loss_function=None,
                 task=None,
                 seq_dependence=None,
                 seq_dep_neg_samples=None,
                 adaptive_method=None,
                 device=-1,  # cpu: -1
                 expid="",
                 log_dir='./logs/',
                 save_path='./checkpoints/'):
        super(MultitaskTrainer, self).__init__(model=model,
                                               optimizer=optimizer,
                                               optimizer_params=optimizer_params,
                                               monitor_mode=monitor_mode,
                                               task=task,
                                               evaluator=evaluator,
                                               device=device,
                                               expid=expid,
                                               log_dir=log_dir,
                                               save_path=save_path)
        if monitor_metric_weight is None:
            monitor_metric_weight = [1, 1]
        if monitor_metric is None:
            monitor_metric = ["AucScore()", "AucScore()"]
        if loss_function is None:
            loss_function = ["BCELoss()", "BCELoss()"]
        if task is None:
            task = ["binary-classification", "binary-classification"]
        if seq_dependence is None:
            seq_dependence = [None, 0]  # task 2 depends sequentially on task 1
        if seq_dep_neg_samples is None:
            seq_dep_neg_samples = [None, "ignore"]  # task 2 ignore negative samples of the task it depends on (task 1)
        self.task = task
        self.seq_dependence = seq_dependence
        self.seq_dep_neg_samples = seq_dep_neg_samples
        self.n_task = len(task)
        self.monitor_metric = [eval(f"metrics.{m_metric}") for m_metric in monitor_metric]
        self.monitor_metric_weight = monitor_metric_weight
        if adaptive_method is not None:
            self.adaptive_method = adaptive_method["name"]
            self.adaptive_params = adaptive_method.get("params", None)
        else:
            self.adaptive_method = "ew"  # equal weighting

        assert self.adaptive_method in ["ew", "uw", "pcgrad", "ple"], f"Adaptive method {self.adaptive_method} not supported."

        self.loss_function = [eval(f"losses.{loss_fn}") if loss_fn.startswith("BPRLoss") or loss_fn.startswith(
                              "FocalLoss") else eval(f"nn.{loss_fn}") for loss_fn in loss_function]

        if self.adaptive_method == "uw":  # uncertainty weighting
            # Initialize parameters for weighting each loss
            self.loss_weight = nn.ParameterList(
                nn.Parameter(torch.ones(1, device=self.device)) for _ in range(self.n_task))
            self.model.add_module("loss weight", self.loss_weight)
        elif self.adaptive_method == "pcgrad":  # project conflicting gradients
            self.optimizer = PCGrad(self.optimizer)
        elif self.adaptive_method == "ple":  # ple adaptive weighting method
            self.loss_weight = self.adaptive_params["initial_weights"].copy()
            self.update_ratios = self.adaptive_params["update_ratios"]
            self.initial_weights = self.adaptive_params["initial_weights"]

    def _train_step(self, features, labels, max_gradient_norm):
        outputs = self.model(features)
        loss, task_losses = self._compute_loss(outputs, labels)
        if self.adaptive_method == "pcgrad":
            loss_list = [task_loss for task_loss in task_losses.values()]
            self.optimizer.pc_backward(loss_list)
        else:
            loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_gradient_norm)
        self.optimizer.step()
        return loss, task_losses

    def _train_one_epoch(self, train_loader, epoch, max_gradient_norm):
        self.model.train()
        total_loss = 0
        total_steps = 0
        task_losses_aggregated = defaultdict(float)  # Aggregate task-specific losses

        # Progress bar with tqdm
        progress_bar = tqdm(train_loader, desc=f"   Training Epoch {epoch + 1}", file=sys.stdout, delay=0.1)

        for features, labels in progress_bar:
            self.optimizer.zero_grad()
            labels = self._get_labels(labels)
            features = self._get_features(features)

            loss, task_losses = self._train_step(features, labels, max_gradient_norm)

            total_loss += loss.item()  # Sum the total loss
            total_steps += 1

            # Aggregate task-specific losses
            for task_name, task_loss in task_losses.items():
                task_losses_aggregated[task_name] += task_loss

            # Update tqdm progress bar
            progress_bar.set_postfix(loss=total_loss / total_steps)

        if self.adaptive_method == "ple":
            for i, initial_weight in enumerate(self.initial_weights):
                self.loss_weight[i] = initial_weight * (self.update_ratios[i] ** epoch)

        avg_loss = total_loss / total_steps

        # Log overall training loss
        self.writer.add_scalar('Loss/train', avg_loss, epoch + 1)

        # Log individual task losses
        for task_name, task_loss in task_losses_aggregated.items():
            avg_task_loss = task_loss / total_steps
            self.writer.add_scalar(f'Loss/train/{task_name}', avg_task_loss, epoch + 1)

        print(f"   Train Loss: {avg_loss:.4f}")
        for task_name, avg_task_loss in task_losses_aggregated.items():
            print(f"   Train {task_name}: {avg_task_loss / total_steps:.4f}")

    def _compute_loss(self, y_pred: {}, y_true: {}):
        loss_list = []
        label_keys = self.model.feature_map["targets"]
        task_losses = {}  # Store individual task losses
        for i in range(len(label_keys)):
            loss_fn = self.loss_function[i]
            if self.seq_dependence[i] is not None:
                task_loss = (
                        self.prev_task_pos_loss(y_pred, y_true, i, label_keys, loss_fn)
                        + self.prev_task_neg_loss(y_pred, y_true, i, label_keys, loss_fn)
                )
            else:
                task_loss = loss_fn(y_pred[f"{label_keys[i]}_pred"], y_true[label_keys[i]])
            loss_list.append(task_loss)
            task_losses[f"Task_{label_keys[i]}_loss"] = task_loss
        if self.adaptive_method == "uw":  # uncertainty weighting
            loss = 0
            for loss_i, w_i in zip(loss_list, self.loss_weight):
                w_i = torch.clamp(w_i, min=0)
                # Compute the weighted loss component for each task
                weighted_loss = 1 / (w_i ** 2) * loss_i
                # Add a regularization term to encourage the learning of useful weights
                regularization = torch.log(1 + w_i ** 2)
                # Sum the weighted loss and the regularization term
                loss += weighted_loss + regularization
        elif self.adaptive_method == "ple":  # ple adaptive weighting method
            loss = 0
            for loss_i, w_i in zip(loss_list, self.loss_weight):
                # Compute the weighted loss component for each task
                weighted_loss = w_i * loss_i
                # Sum the weighted loss and the regularization term
                loss += weighted_loss
        else:  # equal weighting
            loss = torch.sum(torch.stack(loss_list)) / len(loss_list)
        return loss, task_losses

    def prev_task_neg_loss(self, y_pred: {}, y_true: {}, i: int, label_keys: list, loss_fn):
        if self.seq_dep_neg_samples[i] == "ignore":  # ignore negative samples of the task it depends on
            return torch.tensor(0.0, device=y_pred[f"{label_keys[i]}_pred"].device,
                                requires_grad=True)
        elif self.seq_dep_neg_samples[i] == "entropy":  # compute the entropy loss for the negative samples of the task it depends on
            mask = (y_true[label_keys[self.seq_dependence[i]]] == 0).squeeze()
            pred_curr = y_pred[f"{label_keys[i]}_pred"][mask]
            if len(pred_curr) > 0:  # Avoid empty tensors
                if self.task[i] == "multiclass-classification":
                    pseudo_labels = pred_curr.argmax(dim=-1).float()
                    return loss_fn(pred_curr, pseudo_labels)
                elif self.task[i] in ["binary-classification"]:
                    # Entropy loss for pseudo-labeling
                    entropy_loss = -(pred_curr * torch.log(pred_curr + 1e-8) +
                                     (1 - pred_curr) * torch.log(1 - pred_curr + 1e-8))
                    return entropy_loss.mean()
                elif self.task[i] == "regression":
                    # Compute mean and variance for positive samples (clicked articles)
                    positive_mask = (y_true[label_keys[self.seq_dependence[i]]] == 1).squeeze()
                    positive_preds = y_pred[f"{label_keys[i]}_pred"][positive_mask]

                    if len(positive_preds) > 0:
                        mu_pos = positive_preds.mean()
                        sigma_pos = positive_preds.std()

                        # Compute KL divergence loss for non-clicked articles
                        mu_pred = pred_curr.mean()
                        sigma_pred = pred_curr.std()

                        kl_loss = kl_divergence_gaussian(mu_pred, sigma_pred, mu_pos, sigma_pos)
                        return kl_loss
                    else:
                        return torch.tensor(0.0, device=pred_curr.device, requires_grad=True)
                else:
                    raise ValueError(f"Task {self.task[i]} not supported yet for pseudo labeling.")
            else:
                return torch.tensor(0.0, device=y_pred[f"{label_keys[i]}_pred"].device,
                                    requires_grad=True)  # No contribution if no samples
        else:
            raise ValueError(f"{self.seq_dep_neg_samples[i]} not supported!")

    def prev_task_pos_loss(self, y_pred: {}, y_true: {}, i: int, label_keys: list, loss_fn):
        mask = (y_true[label_keys[self.seq_dependence[i]]] == 1).squeeze()  # for each task only consider the positive samples of the task it depends on
        pred_curr = y_pred[f"{label_keys[i]}_pred"][mask]
        true_curr = y_true[label_keys[i]][mask]
        if len(pred_curr) > 0:  # Avoid empty tensors
            return loss_fn(pred_curr, true_curr)
        else:
            return torch.tensor(0.0, device=y_pred[f"{label_keys[i]}_pred"].device,
                                requires_grad=True)  # No contribution if no samples

    def _get_labels(self, labels):
        new_labels = {}
        for i, (label, tensor) in enumerate(labels.items()):
            if self.task[i] == "multiclass-classification":
                new_labels[label] = tensor.to(self.device).long()
            elif self.task[i] in ["binary-classification", "regression"]:
                new_labels[label] = tensor.view(-1, 1).to(self.device).float()
            elif self.task[i] == "ranking":
                new_labels[label] = tensor.to(self.device).float()
            else:
                raise ValueError(f"Task {self.task[i]} not supported.")
        return new_labels

    def _evaluate(self, val_loader, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_steps = 0
            outputs_lists = defaultdict(list)
            labels_lists = defaultdict(list)
            task_losses_aggregated = defaultdict(float)  # Aggregate task-specific losses
            group_id_lists = defaultdict(list)
            label_keys = self.model.feature_map["targets"]
            # Progress bar with tqdm
            progress_bar = tqdm(val_loader, desc=f"   Validating Epoch {epoch + 1}", file=sys.stdout, delay=0.1)
            for features, labels in progress_bar:
                labels = self._get_labels(labels)
                features = self._get_features(features)

                outputs = self.model(features)
                loss, task_losses = self._compute_loss(outputs, labels)
                total_loss += loss.item()  # Sum the loss over the batch
                total_steps += 1

                # Aggregate task-specific losses
                for task_name, task_loss in task_losses.items():
                    task_losses_aggregated[task_name] += task_loss.item()

                for i in range(len(label_keys)):
                    if self.seq_dependence[i] is not None:
                        mask = (labels[label_keys[self.seq_dependence[i]]] == 1).squeeze()  # for each task only consider the positive samples of task it depends on
                        pred_curr = outputs[f"{label_keys[i]}_pred"][mask]
                        true_curr = labels[label_keys[i]][mask]
                        if self.model.feature_map.get("group_id", None):
                            group_id = features[self.model.feature_map["group_id"]][mask]
                            group_id_lists[label_keys[i]].extend(group_id.cpu().numpy().reshape(-1))
                    else:
                        pred_curr = outputs[f"{label_keys[i]}_pred"]
                        true_curr = labels[label_keys[i]]
                        if self.model.feature_map.get("group_id", None):
                            group_id = features[self.model.feature_map["group_id"]]
                            group_id_lists[label_keys[i]].extend(group_id.cpu().numpy().reshape(-1))

                    if self.task[i] == "multiclass-classification":
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy())
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy().reshape(-1))

                    elif self.task[i] in ["binary-classification", "regression"]:
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy().reshape(-1))
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy().reshape(-1))
                    elif self.task[i] == "ranking":
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy())
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy())
                    else:
                        raise ValueError(f"Task {self.task[i]} not supported.")

                # Update tqdm progress bar
                progress_bar.set_postfix(loss=total_loss / total_steps)

            avg_loss = total_loss / total_steps
            # Log validation metrics to TensorBoard
            self.writer.add_scalar('Loss/val', avg_loss, epoch + 1)
            print(f"   Val Loss: {avg_loss:.4f}")
            for task_name, avg_task_loss in task_losses_aggregated.items():
                print(f"   Val {task_name}: {avg_task_loss / total_steps:.4f}")

            # Log individual task losses
            for task_name, task_loss in task_losses_aggregated.items():
                avg_task_loss = task_loss / total_steps
                self.writer.add_scalar(f'Loss/val/{task_name}', avg_task_loss, epoch + 1)

            combined_metric = 0
            for i in range(len(label_keys)):
                # Convert to numpy arrays
                outputs = np.array(outputs_lists[label_keys[i]], np.float64)
                labels = np.array(labels_lists[label_keys[i]], np.float64)
                metric_fn = self.monitor_metric[i]
                metric_weight = self.monitor_metric_weight[i]
                if metric_fn.type == "group_metric" and self.task[i] != "ranking":
                    if self.model.feature_map.get("group_id", None):
                        # Group outputs and labels
                        group_ids = np.array(group_id_lists[label_keys[i]], np.int32)
                        labels, outputs = group_by_id(group_ids, outputs, labels)
                    else:
                        raise ValueError(
                            f"The monitor metric {metric_fn.name} is a group_metric but no group_id is available in the data")

                # Compute the metric
                value = metric_fn(labels, outputs)
                combined_metric += metric_weight * value

                self.writer.add_scalar(f'{metric_fn.name}/val/Task_{label_keys[i]}', value, epoch + 1)
                print(f"   Val Task {label_keys[i]} {metric_fn.name}: {value:.4f}")
            return combined_metric

    def evaluate_test(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            outputs_lists = defaultdict(list)
            labels_lists = defaultdict(list)
            group_id_lists = defaultdict(list)
            label_keys = self.model.feature_map["targets"]
            # Progress bar with tqdm
            progress_bar = tqdm(data_loader, desc=f"   Evaluating Model on Test", file=sys.stdout, delay=0.1)
            for features, labels in progress_bar:
                labels = self._get_labels(labels)
                features = self._get_features(features)

                outputs = self.model(features)

                for i in range(len(label_keys)):
                    if self.seq_dependence[i] is not None:
                        mask = (labels[label_keys[self.seq_dependence[i]]] == 1).squeeze()  # for each task only consider the positive samples of the task it depends on
                        pred_curr = outputs[f"{label_keys[i]}_pred"][mask]
                        true_curr = labels[label_keys[i]][mask]
                        if self.model.feature_map.get("group_id", None):
                            group_id = features[self.model.feature_map["group_id"]][mask]
                            group_id_lists[label_keys[i]].extend(group_id.cpu().numpy().reshape(-1))
                    else:
                        pred_curr = outputs[f"{label_keys[i]}_pred"]
                        true_curr = labels[label_keys[i]]
                        if self.model.feature_map.get("group_id", None):
                            group_id = features[self.model.feature_map["group_id"]]
                            group_id_lists[label_keys[i]].extend(group_id.cpu().numpy().reshape(-1))

                    if self.task[i] == "multiclass-classification":
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy())
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy().reshape(-1))
                    elif self.task[i] in ["binary-classification", "regression"]:
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy().reshape(-1))
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy().reshape(-1))
                    elif self.task[i] == "ranking":
                        outputs_lists[label_keys[i]].extend(pred_curr.data.cpu().numpy())
                        labels_lists[label_keys[i]].extend(true_curr.data.cpu().numpy())
                    else:
                        raise ValueError(f"Task {self.task[i]} not supported.")

            for i in range(len(label_keys)):
                # Convert to numpy arrays
                outputs = np.array(outputs_lists[label_keys[i]], np.float64)
                labels = np.array(labels_lists[label_keys[i]], np.float64)
                group_ids = None
                if self.model.feature_map.get("group_id", None):
                    # Group outputs and labels
                    group_ids = np.array(group_id_lists[label_keys[i]], np.int32)

                evaluation_results = self.evaluator[i].evaluate(labels=labels, predictions=outputs, group_ids=group_ids,
                                                                task=self.task[i])
                print(f"Evaluation Results Task {label_keys[i]} on Test Set:")
                for metric, value in evaluation_results.items():
                    if metric == "Confusion Matrix":
                        print(f"    {metric}: \n{value}")
                    else:
                        print(f"    {metric}: {value:.4f}")
