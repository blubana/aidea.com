"""PCGrad optimizer wrapper for multi-task learning.

This module implements PCGrad (Projecting Conflicting Gradients), introduced in:
Yu et al., 2020, "Gradient Surgery for Multi-Task Learning".

PCGrad modifies per-task gradients before optimizer stepping by projecting away
conflicting components (negative cosine / negative dot-product interactions)
while preserving non-conflicting directions.
"""

from __future__ import annotations

import torch


class PCGrad:
    """Wrap a torch optimizer and apply PCGrad gradient surgery.

    Reference:
        Yu, T., Kumar, S., Gupta, A., Hausman, K., Levine, S., and Finn, C.
        "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020).
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optim = optimizer
        self._task_num: int | None = None

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    def zero_grad(self) -> None:
        self._optim.zero_grad(set_to_none=True)

    def step(self) -> None:
        self._optim.step()

    def pc_backward(self, losses: list[torch.Tensor]) -> None:
        """Backpropagate multiple task losses with PCGrad projection.

        Steps:
        1) Compute per-task gradients.
        2) For each task gradient g_i, project out conflicting component along
           g_j whenever dot(g_i, g_j) < 0.
        3) Sum projected task gradients and write back to parameter `.grad`.
        """
        if not losses:
            return

        self._task_num = len(losses)
        params = [p for group in self._optim.param_groups for p in group["params"]]
        trainable_params = [p for p in params if p.requires_grad]

        per_task_grads: list[torch.Tensor] = []
        param_shapes = [p.shape for p in trainable_params]
        param_numels = [p.numel() for p in trainable_params]

        self.zero_grad()

        for i, loss in enumerate(losses):
            retain_graph = i < (self._task_num - 1)
            loss.backward(retain_graph=retain_graph)

            grads_for_task: list[torch.Tensor] = []
            for p in trainable_params:
                if p.grad is None:
                    grads_for_task.append(torch.zeros_like(p).reshape(-1))
                else:
                    grads_for_task.append(p.grad.detach().clone().reshape(-1))

            per_task_grads.append(torch.cat(grads_for_task, dim=0))
            self.zero_grad()

        projected_grads = [g.clone() for g in per_task_grads]
        eps = torch.finfo(projected_grads[0].dtype).eps

        for i in range(self._task_num):
            g_i = projected_grads[i]
            for j in range(self._task_num):
                if i == j:
                    continue
                # Use the original other-task gradient for conflict detection,
                # while iteratively updating g_i, matching the PCGrad paper.
                g_j = per_task_grads[j]
                dot_ij = torch.dot(g_i, g_j)
                if dot_ij < 0:
                    denom = torch.dot(g_j, g_j)
                    if denom > eps:
                        g_i = g_i - (dot_ij / denom) * g_j
            projected_grads[i] = g_i

        merged_grad = torch.stack(projected_grads, dim=0).sum(dim=0)

        start = 0
        for p, shape, numel in zip(trainable_params, param_shapes, param_numels, strict=True):
            grad_slice = merged_grad[start : start + numel].reshape(shape)
            p.grad = grad_slice.clone()
            start += numel
