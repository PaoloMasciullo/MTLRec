import torch


class DynamicWeightAverage:
    def __init__(self, num_tasks, temperature=2.0):
        """
        Initialize the Dynamic Weight Average (DWA).

        Args:
            num_tasks (int): Number of tasks.
            temperature (float): Temperature parameter for scaling task weights.
        """
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = torch.ones(num_tasks)  # Initialize previous losses for all tasks.

    def update_weights(self, current_losses):
        """
        Update task weights based on the current losses.

        Args:
            current_losses (list or torch.tensor): Current task losses.

        Returns:
            torch.tensor: Updated task weights.
        """
        current_losses = torch.tensor(current_losses)
        # for the first two epochs, the task weights are uniform
        if torch.all(self.prev_losses == torch.ones(self.num_tasks)) or torch.all(self.prev_losses == current_losses):
            task_weights = torch.ones(self.num_tasks)
        else:
            # Compute the relative rate of change for each task loss.
            rates_of_change = current_losses / (self.prev_losses + 1e-8)

            # Update weights using the softmax formula.
            exp_weights = torch.exp(rates_of_change / self.temperature)
            task_weights = self.num_tasks * exp_weights / torch.sum(exp_weights)

        # Update previous losses for the next iteration.
        self.prev_losses = current_losses
        return task_weights


class PCGrad:
    def __init__(self, optimizer, reduction='mean', use_mixed_precision=False):
        self._optim = optimizer
        self._reduction = reduction
        self._use_mixed_precision = use_mixed_precision

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        """Clear the gradient of the parameters."""
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        """Update the parameters with the gradient."""
        self._optim.step()

    def pc_backward(self, objectives):
        """Compute and apply gradients with PCGrad."""
        grads, shared_mask = self._compute_gradients(objectives)
        pc_grad = self._project_conflicting(grads, shared_mask)
        self._apply_gradients(pc_grad)

    def _compute_gradients(self, objectives):
        """Compute gradients for all objectives."""
        self.zero_grad()

        # Perform a combined backward pass
        torch.autograd.backward(objectives, [torch.ones_like(obj) for obj in objectives])

        grads = []
        for group in self._optim.param_groups:
            group_grads = []
            for param in group['params']:
                if param.grad is not None:
                    group_grads.append(param.grad.view(-1))
                else:
                    group_grads.append(torch.zeros(param.numel(), device=param.device))
            grads.append(torch.cat(group_grads))

        # Stack gradients into a single tensor for efficiency
        grad_stack = torch.stack(grads, dim=0)  # Shape: (num_tasks, total_params)

        # Compute shared mask across tasks
        shared_mask = (grad_stack != 0).all(dim=0)

        return grad_stack, shared_mask

    def _project_conflicting(self, grad_stack, shared_mask):
        """Resolve gradient conflicts using a highly vectorized implementation."""
        num_tasks, total_params = grad_stack.shape

        # Compute dot products and gradient norms
        dot_products = torch.matmul(grad_stack, grad_stack.T)  # Shape: (num_tasks, num_tasks)
        grad_norms = grad_stack.norm(dim=1, keepdim=True).clamp(min=1e-8)  # Avoid division by zero

        # Compute projection factors using einsum for efficiency
        projection_factors = torch.einsum(
            'ij,ij->ij',
            dot_products,
            (dot_products < 0).float(),  # Only negative dot products contribute
        ) / (grad_norms @ grad_norms.T)

        # Apply projection adjustments
        adjustments = torch.einsum('ij,jk->ik', projection_factors, grad_stack)  # Shape: (num_tasks, total_params)
        projected_grads = grad_stack - adjustments

        # Aggregate gradients based on the reduction method
        merged_grad = torch.zeros(total_params, device=grad_stack.device)
        if self._reduction == 'mean':
            merged_grad[shared_mask] = projected_grads[:, shared_mask].mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared_mask] = projected_grads[:, shared_mask].sum(dim=0)
        else:
            raise ValueError("Invalid reduction method")

        return merged_grad

    def _apply_gradients(self, merged_grad):
        """Set the modified gradients back to the model parameters."""
        offset = 0
        for group in self._optim.param_groups:
            for param in group['params']:
                grad_size = param.numel()
                if param.grad is not None:
                    param.grad.copy_(merged_grad[offset:offset + grad_size].view_as(param))
                offset += grad_size


class DynamicTaskPriorityWeighting:
    def __init__(self, n_tasks, alpha=1.0, beta=1.0):
        """
        Initializes the dynamic task priority weighting class.
        Args:
            n_tasks (int): Number of tasks.
            alpha (float): Weight for task relevance factor.
            beta (float): Weight for task progress factor.
        """
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.beta = beta
        self.task_losses = [[] for _ in range(n_tasks)]
        self.task_gradients = [[] for _ in range(n_tasks)]
        self.max_history = 2  # Limit history to the last two entries for both losses and gradients.

    def compute_task_progress(self, task_index):
        """
        Computes the progress factor for a task based on recent loss changes.
        Args:
            task_index (int): The index of the task.
        Returns:
            float: The progress factor (higher is better progress).
        """
        if len(self.task_losses[task_index]) < 2:
            return 1.0  # Default value when not enough history is available.

        # Compute the absolute change in loss between the last two updates.
        recent_losses = self.task_losses[task_index]
        delta_loss = abs(recent_losses[-1] - recent_losses[-2])
        return 1 / (1 + delta_loss)  # Higher progress if the loss change is small.

    @staticmethod
    def compute_task_relevance(task_gradients):
        """
        Computes the relevance factor for a task based on gradient similarity.
        Args:
            task_gradients (list): A list of gradient tensors for the task.
        Returns:
            float: The relevance factor (higher indicates more relevance).
        """
        if len(task_gradients) < 2:
            return 0.5  # Default relevance when not enough gradients are available.

        # Calculate the cosine similarity between the last two gradients.
        grad_1, grad_2 = task_gradients[-2:]
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_1.view(-1), grad_2.view(-1), dim=0
        ).item()

        # Map cosine similarity from [-1, 1] to [0, 1] range.
        return (cos_sim + 1) / 2

    def compute_task_weights(self):
        """
        Computes dynamic weights for all tasks based on progress and relevance factors.
        Returns:
            torch.Tensor: Normalized weights for all tasks.
        """
        # Calculate progress and relevance factors for each task.
        progress_factors = [self.compute_task_progress(i) for i in range(self.n_tasks)]
        relevance_factors = [self.compute_task_relevance(self.task_gradients[i]) for i in range(self.n_tasks)]

        # Combine the factors with alpha and beta scaling.
        combined_factors = [
            self.alpha * relevance + self.beta * progress
            for relevance, progress in zip(relevance_factors, progress_factors)
        ]

        # Normalize the combined factors to obtain the weights.
        total = sum(combined_factors)
        if total == 0:  # Handle the edge case where total is zero.
            return torch.ones(self.n_tasks, dtype=torch.float32) / self.n_tasks

        return torch.tensor(
            [factor / total for factor in combined_factors],
            dtype=torch.float32
        )

    def update_task_tracking(self, task_index, loss, gradient):
        """
        Updates the loss and gradient history for a specific task.
        Args:
            task_index (int): The index of the task to update.
            loss (torch.Tensor): The current loss value for the task.
            gradient (torch.Tensor): The gradient tensor for the task.
        """
        # Append the new loss and gradient to the history.
        self.task_losses[task_index].append(loss.item())
        self.task_gradients[task_index].append(gradient.clone().detach())

        # Retain only the last `max_history` entries.
        self.task_losses[task_index] = self.task_losses[task_index][-self.max_history:]
        self.task_gradients[task_index] = self.task_gradients[task_index][-self.max_history:]
