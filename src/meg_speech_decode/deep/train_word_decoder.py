from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import WordDecoderConfig
from .datasets import SentenceWordDataset, collate_sentences
from .losses import SigLIPLoss
from .models import BrainWordCNN, SentenceContextEncoder, T5WordEncoder


def _bucket_split(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 10
    if h < 8:
        return "train"
    if h == 8:
        return "val"
    return "test"


def _build_splits(dataset: SentenceWordDataset) -> tuple[list[int], list[int], list[int]]:
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for i, s in enumerate(dataset.samples):
        bucket = _bucket_split(s.sentence_key)
        if bucket == "train":
            train_idx.append(i)
        elif bucket == "val":
            val_idx.append(i)
        else:
            test_idx.append(i)
    return train_idx, val_idx, test_idx


def _flatten_batch(words_nested: list[list[str]], mask: torch.Tensor) -> tuple[list[str], torch.Tensor]:
    flat_words: list[str] = []
    positions: list[int] = []
    batch, seq = mask.shape
    for b in range(batch):
        for t in range(seq):
            if not bool(mask[b, t].item()):
                flat_words.append(words_nested[b][t])
                positions.append(b * seq + t)
    return flat_words, torch.tensor(positions, dtype=torch.long)


def _dedupe_by_word(words: list[str], meg_emb: torch.Tensor, text_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    keep = []
    seen = set()
    for i, w in enumerate(words):
        key = w.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    idx = torch.tensor(keep, device=meg_emb.device, dtype=torch.long)
    return meg_emb.index_select(0, idx), text_emb.index_select(0, idx)


def _topk_for_candidates(pred_emb: torch.Tensor, cand_emb: torch.Tensor, target_ids: torch.Tensor, k: int) -> float:
    pred = torch.nn.functional.normalize(pred_emb, dim=-1)
    cand = torch.nn.functional.normalize(cand_emb, dim=-1)
    logits = pred @ cand.t()
    topk = logits.topk(k=min(k, logits.shape[1]), dim=1).indices
    correct = (topk == target_ids.unsqueeze(1)).any(dim=1).float().mean()
    return float(correct.item())


def _balanced_recall_at_10(
    pred_emb: torch.Tensor,
    words: list[str],
    vocab: list[str],
    vocab_emb: torch.Tensor,
    top_words: set[str],
) -> float:
    by_word: dict[str, list[int]] = {}
    for i, w in enumerate(words):
        wk = w.lower().strip()
        if wk in top_words:
            by_word.setdefault(wk, []).append(i)

    if not by_word:
        return 0.0

    id_map = {w: i for i, w in enumerate(vocab)}
    recalls = []
    for w, idxs in by_word.items():
        idx_t = torch.tensor(idxs, device=pred_emb.device, dtype=torch.long)
        targets = torch.full((len(idxs),), id_map[w], device=pred_emb.device, dtype=torch.long)
        score = _topk_for_candidates(pred_emb.index_select(0, idx_t), vocab_emb, targets, k=10)
        recalls.append(score)
    return float(np.mean(recalls))


def run_word_decoder_training(config: WordDecoderConfig) -> dict[str, float]:
    dataset = SentenceWordDataset(
        manifest_csv=config.manifest_csv,
        meg_sfreq=config.meg_sfreq,
        l_freq=config.l_freq,
        h_freq=config.h_freq,
        window_s=config.window_s,
        baseline_s=config.baseline_s,
        max_words_per_sentence=config.max_words_per_sentence,
    )
    train_idx, val_idx, test_idx = _build_splits(dataset)
    if not train_idx or not val_idx:
        raise RuntimeError("Split failed: need non-empty train and val sentence partitions")

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_bs = max(1, min(config.batch_size, len(train_ds)))
    val_bs = max(1, min(config.batch_size, len(val_ds)))
    test_bs = max(1, min(config.batch_size, len(test_ds))) if len(test_ds) > 0 else 1

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_sentences)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, collate_fn=collate_sentences)
    test_loader = DataLoader(test_ds, batch_size=test_bs, shuffle=False, collate_fn=collate_sentences)

    sample = dataset[0]
    channels = int(sample["meg"].shape[1])

    train_words = [ev.word.lower().strip() for i in train_idx for ev in dataset.samples[i].events]
    vocab_counts = Counter(train_words)
    vocab = sorted(vocab_counts.keys())
    top_words = {w for w, _ in vocab_counts.most_common(config.vocab_eval_size)}

    subjects = sorted({s.subject_id for s in dataset.samples})
    subject_to_id = {s: i for i, s in enumerate(subjects)}

    device = torch.device(config.device)
    brain = BrainWordCNN(
        in_channels=channels,
        hidden_dim=config.cnn_hidden_dim,
        out_dim=config.proj_dim,
        num_subjects=len(subject_to_id),
    ).to(device)
    sentence = SentenceContextEncoder(
        dim=config.proj_dim,
        layers=config.sentence_layers,
        heads=config.sentence_heads,
        dropout=config.sentence_dropout,
    ).to(device)
    text_encoder = T5WordEncoder(model_name=config.t5_model_name)
    text_encoder.to(device)
    text_encoder.eval()

    criterion = SigLIPLoss(
        temperature_init=config.temperature_init,
        bias_init=config.bias_init,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(brain.parameters()) + list(sentence.parameters()) + list(criterion.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    vocab_emb = text_encoder(vocab, device=device)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        brain.train(True)
        sentence.train(True)
        tr_losses: list[float] = []

        for batch in train_loader:
            meg = batch["meg"].to(device)
            pad_mask = batch["padding_mask"].to(device)
            words_nested = batch["words_nested"]
            subject_ids = torch.tensor([subject_to_id[s] for s in batch["subject_ids"]], device=device, dtype=torch.long)

            bsz, seq, ch, t = meg.shape
            meg_flat = meg.view(bsz * seq, ch, t)
            subject_flat = subject_ids.unsqueeze(1).repeat(1, seq).reshape(-1)

            word_repr = brain(meg_flat, subject_flat).view(bsz, seq, -1)
            ctx_repr = sentence(word_repr, padding_mask=pad_mask)

            flat_words, flat_pos = _flatten_batch(words_nested, pad_mask.detach().cpu())
            pred = ctx_repr.view(bsz * seq, -1).index_select(0, flat_pos.to(device))
            tgt = text_encoder(flat_words, device=device)

            pred, tgt = _dedupe_by_word(flat_words, pred, tgt)
            if pred.shape[0] < 2:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, tgt)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        brain.train(False)
        sentence.train(False)
        va_losses: list[float] = []
        va_r10_balanced: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                meg = batch["meg"].to(device)
                pad_mask = batch["padding_mask"].to(device)
                words_nested = batch["words_nested"]
                subject_ids = torch.tensor([subject_to_id[s] for s in batch["subject_ids"]], device=device, dtype=torch.long)

                bsz, seq, ch, t = meg.shape
                meg_flat = meg.view(bsz * seq, ch, t)
                subject_flat = subject_ids.unsqueeze(1).repeat(1, seq).reshape(-1)
                word_repr = brain(meg_flat, subject_flat).view(bsz, seq, -1)
                ctx_repr = sentence(word_repr, padding_mask=pad_mask)

                flat_words, flat_pos = _flatten_batch(words_nested, pad_mask.detach().cpu())
                pred = ctx_repr.view(bsz * seq, -1).index_select(0, flat_pos.to(device))
                tgt = text_encoder(flat_words, device=device)
                pred_d, tgt_d = _dedupe_by_word(flat_words, pred, tgt)
                if pred_d.shape[0] >= 2:
                    va_losses.append(float(criterion(pred_d, tgt_d).item()))

                va_r10_balanced.append(
                    _balanced_recall_at_10(
                        pred_emb=pred,
                        words=flat_words,
                        vocab=vocab,
                        vocab_emb=vocab_emb,
                        top_words=top_words,
                    )
                )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(tr_losses)) if tr_losses else 0.0,
                "val_dsiglip_loss": float(np.mean(va_losses)) if va_losses else 0.0,
                "val_balanced_recall_at_10": float(np.mean(va_r10_balanced)) if va_r10_balanced else 0.0,
            }
        )

    test_r10_balanced: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            meg = batch["meg"].to(device)
            pad_mask = batch["padding_mask"].to(device)
            words_nested = batch["words_nested"]
            subject_ids = torch.tensor([subject_to_id[s] for s in batch["subject_ids"]], device=device, dtype=torch.long)

            bsz, seq, ch, t = meg.shape
            meg_flat = meg.view(bsz * seq, ch, t)
            subject_flat = subject_ids.unsqueeze(1).repeat(1, seq).reshape(-1)
            word_repr = brain(meg_flat, subject_flat).view(bsz, seq, -1)
            ctx_repr = sentence(word_repr, padding_mask=pad_mask)

            flat_words, flat_pos = _flatten_batch(words_nested, pad_mask.detach().cpu())
            pred = ctx_repr.view(bsz * seq, -1).index_select(0, flat_pos.to(device))
            test_r10_balanced.append(
                _balanced_recall_at_10(
                    pred_emb=pred,
                    words=flat_words,
                    vocab=vocab,
                    vocab_emb=vocab_emb,
                    top_words=top_words,
                )
            )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.output_dir / "word_decoder_dascoli.pt"
    hist_path = config.output_dir / "word_decoder_history.json"
    metrics_path = config.output_dir / "word_decoder_metrics.json"

    torch.save(
        {
            "config": config.__dict__,
            "subject_to_id": subject_to_id,
            "vocab": vocab,
            "brain": brain.state_dict(),
            "sentence": sentence.state_dict(),
            "criterion": criterion.state_dict(),
        },
        ckpt_path,
    )
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    final = history[-1]
    metrics = {
        "val_dsiglip_loss": final["val_dsiglip_loss"],
        "val_balanced_recall_at_10": final["val_balanced_recall_at_10"],
        "test_balanced_recall_at_10": float(np.mean(test_r10_balanced)) if test_r10_balanced else 0.0,
        "train_sentences": len(train_idx),
        "val_sentences": len(val_idx),
        "test_sentences": len(test_idx),
        "vocab_eval_size": min(config.vocab_eval_size, len(top_words)),
        "checkpoint": str(ckpt_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
