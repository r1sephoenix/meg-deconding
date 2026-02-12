from __future__ import annotations

import torch
import torch.nn as nn


class MegEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, depth: int, n_heads: int, proj_dim: int) -> None:
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.proj = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frontend(x)
        z = z.transpose(1, 2)
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.proj(z)


class Wav2VecTargetEncoder(nn.Module):
    def __init__(self, proj_dim: int) -> None:
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio is required for MEGFormer pipeline") from exc

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        self.wav2vec = bundle.get_model()
        for p in self.wav2vec.parameters():
            p.requires_grad = False
        self.wav2vec.eval()
        self.proj = nn.Linear(768, proj_dim)

    @torch.no_grad()
    def _extract(self, audio: torch.Tensor) -> torch.Tensor:
        features, _ = self.wav2vec.extract_features(audio)
        last = features[-1]
        return last.mean(dim=1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self._extract(audio)
        return self.proj(z)


class MegFormerLike(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, depth: int, n_heads: int, proj_dim: int) -> None:
        super().__init__()
        self.meg_encoder = MegEncoder(in_channels, hidden_dim, depth, n_heads, proj_dim)
        self.audio_encoder = Wav2VecTargetEncoder(proj_dim=proj_dim)

    def forward(self, meg: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        meg_emb = self.meg_encoder(meg)
        audio_emb = self.audio_encoder(audio)
        return meg_emb, audio_emb


class SubjectConditioning(nn.Module):
    def __init__(self, num_subjects: int, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_subjects, channels)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        bias = self.embedding(subject_ids).unsqueeze(-1)
        return x + bias


class BrainWordCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, num_subjects: int) -> None:
        super().__init__()
        self.subject_cond = SubjectConditioning(num_subjects=num_subjects, channels=in_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        x = self.subject_cond(x, subject_ids)
        z = self.net(x).squeeze(-1)
        return self.proj(z)


class SentenceContextEncoder(nn.Module):
    def __init__(self, dim: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=padding_mask)


class T5WordEncoder(nn.Module):
    def __init__(self, model_name: str = "t5-large") -> None:
        super().__init__()
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for T5 word embeddings") from exc

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def forward(self, words: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(words, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        out = self.model(**tokens, output_hidden_states=True)
        hidden_states = out.hidden_states
        mid = len(hidden_states) // 2
        emb = hidden_states[mid].mean(dim=1)
        return emb
