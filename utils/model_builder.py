from pathlib import Path
from typing import Optional

import torch

from dataset.dataset import create_imagenet_split
from utils.logging import WandbLogger
from utils.misc import EasyDict, sanitize_model_config, sanitize_train_config


def _build_optimi_optimizer(model, optimizer_config, initial_lr: float):
    from optimi import AdamW as OptimiAdamW, StableAdamW as OptimiStableAdamW
    from optimi.utils import param_groups_weight_decay

    optimizer_type = str(optimizer_config.get("optimizer_type", "optimi_adamw")).lower()
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    no_weight_decay_layers = list(optimizer_config.get("no_weight_decay_layers", []))
    betas = (float(optimizer_config.adam_b1), float(optimizer_config.adam_b2))
    gradient_release = bool(optimizer_config.get("gradient_release", False))

    params = param_groups_weight_decay(model, weight_decay=weight_decay, additional_layers=no_weight_decay_layers)

    kwargs = dict(lr=initial_lr, betas=betas, gradient_release=gradient_release)
    triton = optimizer_config.get("triton", None)
    if triton is not None:
        kwargs["triton"] = bool(triton)

    cls = OptimiStableAdamW if optimizer_type == "optimi_stableadamw" else OptimiAdamW
    return cls(params, **kwargs)


def create_learning_rate_fn(
    learning_rate,
    warmup_steps,
    total_steps,
    lr_schedule="const",
):
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)
    learning_rate = float(learning_rate)

    def _warmup(step: int) -> float:
        if step >= warmup_steps:
            return learning_rate
        alpha = float(step) / float(warmup_steps)
        return 1e-6 + alpha * (learning_rate - 1e-6)

    if lr_schedule in ["cosine", "cos"]:
        cosine_steps = max(total_steps - warmup_steps, 1)

        def _schedule(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step)
            t = min(step - warmup_steps, cosine_steps)
            cos = 0.5 * (1.0 + torch.cos(torch.tensor(t / cosine_steps * torch.pi)).item())
            return learning_rate * ((1.0 - 1e-6) * cos + 1e-6)

    elif lr_schedule == "const":

        def _schedule(step: int) -> float:
            if step < warmup_steps:
                return _warmup(step)
            return learning_rate

    else:
        raise NotImplementedError(lr_schedule)

    return _schedule


def build_model_dict(config, model_class, *, workdir: str = "runs"):
    print("Building model...")
    model_config = sanitize_model_config(config.model)
    model = model_class(
        num_classes=config.dataset.num_classes,
        **model_config,
    )

    print("Building dataset...")
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()

    batch_size_per_node = config.dataset.batch_size // world_size
    resolution = int(config.dataset.resolution)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))

    train_loader, preprocess_fn, postprocess_fn = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=batch_size_per_node,
        split="train",
        **config.dataset.kwargs,
    )

    eval_loader, _, _ = create_imagenet_split(
        resolution=resolution,
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        batch_size=config.dataset.eval_batch_size // world_size,
        split="val",
        **config.dataset.kwargs,
    )

    learning_rate_fn = create_learning_rate_fn(**config.optimizer.lr_schedule)

    optimizer_type = str(config.optimizer.get("optimizer_type", "pytorch_adamw")).lower()
    low_precision_model = bool(config.optimizer.get("low_precision_model", False)) and optimizer_type != "pytorch_adamw"
    if low_precision_model:
        from optimi.utils import to_low_precision
        from models.generator import RotaryEmbedding
        fp32_modules = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.Embedding, RotaryEmbedding)
        to_low_precision(model, dtype=torch.bfloat16, fp32_modules=fp32_modules)

    if optimizer_type == "pytorch_adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate_fn(0),
            weight_decay=float(config.optimizer.get("weight_decay", 0.0)),
            betas=(float(config.optimizer.adam_b1), float(config.optimizer.adam_b2)),
        )
    else:
        optimizer = _build_optimi_optimizer(model, config.optimizer, initial_lr=learning_rate_fn(0))
        if bool(config.optimizer.get("gradient_release", False)):
            from optimi import prepare_for_gradient_release
            prepare_for_gradient_release(model, optimizer)

    logger = WandbLogger()
    w_cfg = EasyDict(dict(config.get("logging", {})))
    use_wandb = bool(w_cfg.get("use_wandb", config.get("use_wandb", True)))
    if "use_wandb" in w_cfg:
        del w_cfg["use_wandb"]
    output_root = Path(workdir).resolve()
    logger.set_logging(
        config=config,
        use_wandb=use_wandb,
        workdir=str(output_root),
        **w_cfg,
    )

    return EasyDict(
        model=model,
        optimizer=optimizer,
        logger=logger,
        eval_loader=eval_loader,
        train_loader=train_loader,
        dataset_name=f"imagenet{resolution}",
        preprocess_fn=preprocess_fn,
        postprocess_fn=postprocess_fn,
        train=sanitize_train_config(config.train),
        learning_rate_fn=learning_rate_fn,
        feature=config.get("feature", {}),
        gradient_release=bool(config.optimizer.get("gradient_release", False)) and optimizer_type != "pytorch_adamw",
        skip_grad_clip=optimizer_type == "optimi_stableadamw",
    )
