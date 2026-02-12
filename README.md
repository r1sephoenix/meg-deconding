# MEG Decoding

Проект с тремя пайплайнами для декодирования речи/лексики из MEG.

## Пайплайны
- `baseline`: Ridge-регрессия `MEG -> mel spectrogram`
- `train-megformer`: MEGFormer-like contrastive pipeline (`MEG embedding -> wav2vec2 embedding`)
- `train-word-decoder`: sentence-level lexical decoder

## Установка
```bash
uv sync --extra deep
```

## 1) Baseline

Manifest (`CSV`):
- `meg_path`
- `audio_path`
- `start_s`
- `end_s`

```bash
uv run python scripts/prepare_baseline_data.py \
  --audio-dir "" \
  --meg-dir "" \
  --manifest-output "" \
  --strict
```

Ручные шаги (если нужно запускать по отдельности):
```bash
uv run python scripts/fetch_drive_meg.py \
  --output-dir "artifacts/raw_meg" \
  --strict

uv run python scripts/build_manifest.py \
  --meg-dir "artifacts/raw_meg" \
  --audio-dir ""
  --output "artifacts/manifest.csv"

uv run meg-decode baseline --config "configs/baseline.yaml"
```

## 2) MEGFormer (contrastive)

```bash
uv run meg-decode train-megformer --config "configs/deep/megformer.yaml"
```

## 3) Word Decoder 

Реализовано в пайплайне:
- окно 3 сек от onset слова
- baseline correction по первым 0.5 сек
- sentence-level контекстный энкодер
- target embeddings из среднего слоя `t5-large`
- D-SigLIP лосс с дедупликацией повторов слов в батче
- deterministic split 80/10/10 по `sentence_key`
- метрика `balanced recall@10` на top-`vocab_eval_size` слов (по умолчанию 250)

### Формат word manifest
Обязательные колонки:
- `meg_path`
- `word`
- `start_s`
- `end_s`
- `sentence_id`
- `subject_id`

Опционально:
- `sentence_text` (используется как стабильный ключ для split)
- `word_index` (если нет, проставится автоматически)

Подготовка:
```bash
uv run python scripts/build_word_manifest.py \
  --input "PATH/TO/RAW_WORD_EVENTS.tsv" \
  --output "artifacts/word_manifest.csv"
```

Обучение:
```bash
uv run meg-decode train-word-decoder --config "configs/deep/word_decoder.yaml"
```

## Проверка
```bash
uv run python -m compileall src scripts
```
