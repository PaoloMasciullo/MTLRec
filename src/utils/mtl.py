import torch


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
