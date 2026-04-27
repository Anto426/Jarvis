import math
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain

import torch
from torch.optim import Optimizer


class CPUAdamW(Optimizer):
    """AdamW with optimizer state kept on CPU to reduce GPU memory pressure."""

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def _cast_state_value_to_cpu(self, value, key=None):
        if key == "step":
            if isinstance(value, torch.Tensor):
                return int(value.detach().cpu().item())
            return int(value)
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.detach().to("cpu", dtype=torch.float32)
            return value.detach().to("cpu")
        if isinstance(value, dict):
            return {
                item_key: self._cast_state_value_to_cpu(item_value, key=item_key)
                for item_key, item_value in value.items()
            }
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return type(value)(
                self._cast_state_value_to_cpu(item_value) for item_value in value
            )
        return value

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")

        param_lens = (len(group["params"]) for group in groups)
        saved_lens = (len(group["params"]) for group in saved_groups)
        if any(
            group_len != saved_len
            for group_len, saved_len in zip(param_lens, saved_lens, strict=True)
        ):
            raise ValueError(
                "loaded state dict contains a parameter group that does not match "
                "the size of optimizer's group"
            )

        id_map = dict(
            zip(
                chain.from_iterable(group["params"] for group in saved_groups),
                chain.from_iterable(group["params"] for group in groups),
                strict=True,
            )
        )

        state = defaultdict(dict)
        for key, value in state_dict["state"].items():
            if key in id_map:
                state[id_map[key]] = self._cast_state_value_to_cpu(value)
            else:
                state[key] = self._cast_state_value_to_cpu(value)

        def update_group(group, new_group):
            new_group["params"] = group["params"]
            if "param_names" in group and "param_names" not in new_group:
                new_group["param_names"] = group["param_names"]
            return new_group

        param_groups = [
            update_group(group, new_group)
            for group, new_group in zip(groups, saved_groups, strict=True)
        ]
        self.__setstate__({"state": state, "param_groups": param_groups})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("CPUAdamW does not support sparse gradients.")

                grad_cpu = param.grad.detach().to("cpu", dtype=torch.float32)
                param_cpu = param.detach().to("cpu", dtype=torch.float32)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        param_cpu,
                        dtype=torch.float32,
                        device="cpu",
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        param_cpu,
                        dtype=torch.float32,
                        device="cpu",
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                if weight_decay != 0:
                    param_cpu.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)

                param_cpu.addcdiv_(exp_avg, denom, value=-step_size)
                param.copy_(param_cpu.to(device=param.device, dtype=param.dtype))

        return loss
