from __future__ import annotations

import argparse

from .config import PipelineConfig
from .deep.config import MegformerConfig, WordDecoderConfig
from .deep.train_megformer import run_megformer_training
from .deep.train_word_decoder import run_word_decoder_training
from .pipeline import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="MEG speech decoding pipelines")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_base = sub.add_parser("baseline", help="Run ridge baseline MEG->mel")
    p_base.add_argument("--config", required=True, help="Path to baseline YAML config")

    p_megf = sub.add_parser("train-megformer", help="Train MEGFormer-like CLIP pipeline")
    p_megf.add_argument("--config", required=True, help="Path to MEGFormer YAML config")

    p_word = sub.add_parser("train-word-decoder", help="Train lexical decoder from MEG")
    p_word.add_argument("--config", required=True, help="Path to word decoder YAML config")

    args = parser.parse_args()

    if args.cmd == "baseline":
        config = PipelineConfig.from_yaml(args.config)
        metrics = run_baseline(config)
        print(f"Baseline done. MSE={metrics['mse']:.4f} R2={metrics['r2']:.4f}")
        return

    if args.cmd == "train-megformer":
        config = MegformerConfig.from_yaml(args.config)
        metrics = run_megformer_training(config)
        print(
            "MEGFormer-like done. "
            f"val_loss={metrics['val_loss']:.4f} val_recall@10={metrics['val_recall_at_10']:.4f}"
        )
        return

    if args.cmd == "train-word-decoder":
        config = WordDecoderConfig.from_yaml(args.config)
        metrics = run_word_decoder_training(config)
        print(
            "Word decoder done. "
            f"val_dsiglip_loss={metrics['val_dsiglip_loss']:.4f} "
            f"val_balanced_recall@10={metrics['val_balanced_recall_at_10']:.4f} "
            f"test_balanced_recall@10={metrics['test_balanced_recall_at_10']:.4f}"
        )
        return

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
