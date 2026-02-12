from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import MegformerConfig
from .datasets import MegAudioPairDataset
from .losses import clip_loss, retrieval_at_k
from .models import MegFormerLike


def _split_indices(n: int, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_size)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def _run_epoch(
    model: MegFormerLike,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    temperature: float,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    accs: list[float] = []

    for meg, audio in loader:
        meg = meg.to(device)
        audio = audio.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            meg_emb, audio_emb = model(meg, audio)
            loss = clip_loss(meg_emb, audio_emb, temperature=temperature)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        accs.append(retrieval_at_k(meg_emb.detach(), audio_emb.detach(), k=10))

    return float(np.mean(losses)), float(np.mean(accs))


def run_megformer_training(config: MegformerConfig) -> dict[str, float]:
    dataset = MegAudioPairDataset(
        manifest_csv=config.manifest_csv,
        meg_sfreq=config.meg_sfreq,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        segment_s=config.segment_s,
    )
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples for contrastive training")

    train_idx, val_idx = _split_indices(len(dataset), config.val_size, config.random_seed)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    train_bs = max(1, min(config.batch_size, len(train_ds)))
    val_bs = max(1, min(config.batch_size, len(val_ds)))
    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False)

    sample_meg, _ = dataset[0]
    in_channels = sample_meg.shape[0]
    device = torch.device(config.device)

    model = MegFormerLike(
        in_channels=in_channels,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        n_heads=config.n_heads,
        proj_dim=config.proj_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.meg_encoder.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        tr_loss, tr_r10 = _run_epoch(model, train_loader, device, optimizer, config.temperature)
        va_loss, va_r10 = _run_epoch(model, val_loader, device, None, config.temperature)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr_loss,
                "train_recall_at_10": tr_r10,
                "val_loss": va_loss,
                "val_recall_at_10": va_r10,
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.output_dir / "megformer_like.pt"
    hist_path = config.output_dir / "megformer_history.json"
    final_path = config.output_dir / "megformer_metrics.json"

    torch.save({"config": config.__dict__, "state_dict": model.state_dict()}, ckpt_path)
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final = history[-1]
    metrics = {
        "val_loss": final["val_loss"],
        "val_recall_at_10": final["val_recall_at_10"],
        "checkpoint": str(ckpt_path),
    }
    final_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
