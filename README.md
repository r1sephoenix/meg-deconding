# MEG Speech Decoding

Проект с тремя пайплайнами для декодирования речи/лексики из MEG.

## Пайплайны
- `baseline`: Ridge-регрессия `MEG -> mel spectrogram`
- `train-megformer`: MEGFormer-like contrastive pipeline (`MEG embedding -> wav2vec2 embedding`)
- `train-word-decoder`: sentence-level lexical decoder, адаптированный под протокол статьи Nature `s41467-025-65499-0`

## Установка
```bash
cd "/Users/ilyamikheev/Documents/New project"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e .[deep]
```

## 1) Baseline (MEG -> mel)

Manifest (`CSV`):
- `meg_path`
- `audio_path`
- `start_s`
- `end_s`

Подготовка данных одной командой (скачать `.fif` из Google Drive + собрать `manifest.csv`):
```bash
python scripts/prepare_baseline_data.py \
  --audio-dir "/Users/ilyamikheev/Downloads/drive-download-20241028T122739Z-001 2" \
  --meg-dir "/Users/ilyamikheev/Documents/New project/artifacts/raw_meg" \
  --manifest-output "/Users/ilyamikheev/Documents/New project/artifacts/manifest.csv" \
  --strict
```

Ручные шаги (если нужно запускать по отдельности):
```bash
python scripts/fetch_drive_meg.py \
  --output-dir "/Users/ilyamikheev/Documents/New project/artifacts/raw_meg" \
  --strict

python scripts/build_manifest.py \
  --meg-dir "/Users/ilyamikheev/Documents/New project/artifacts/raw_meg" \
  --audio-dir "/Users/ilyamikheev/Downloads/drive-download-20241028T122739Z-001 2" \
  --output "/Users/ilyamikheev/Documents/New project/artifacts/manifest.csv"

meg-decode baseline --config "/Users/ilyamikheev/Documents/New project/configs/baseline.yaml"
```

## 2) MEGFormer-like (contrastive)

```bash
meg-decode train-megformer --config "/Users/ilyamikheev/Documents/New project/configs/deep/megformer.yaml"
```

## 3) Word Decoder (адаптация под `s41467-025-65499-0`)

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
python scripts/build_word_manifest.py \
  --input "/PATH/TO/RAW_WORD_EVENTS.tsv" \
  --output "/Users/ilyamikheev/Documents/New project/artifacts/word_manifest.csv"
```

Обучение:
```bash
meg-decode train-word-decoder --config "/Users/ilyamikheev/Documents/New project/configs/deep/word_decoder.yaml"
```

Артефакты:
- `artifacts/word_decoder/word_decoder_dascoli.pt`
- `artifacts/word_decoder/word_decoder_history.json`
- `artifacts/word_decoder/word_decoder_metrics.json`

## Примечание по репликации
Пайплайн выровнен по ключевым компонентам статьи, но это не bitwise-реплика исходного приватного кода/всех датасет-специфичных деталей.

## Проверка
```bash
python3 -m compileall src scripts
```
