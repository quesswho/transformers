# Transformers

Pytorch implementation of decoder transformers

Includes a next token predictor in examples/generative_lm

## Setup

Install the dependencies into your Python 3.12 environment:

```powershell
pip install torch numpy tokenizers transformers
```

The tokenizer is a SentencePiece-style Unigram model trained with HuggingFace
`tokenizers` and wrapped in a `PreTrainedTokenizerFast`, so a trained tokenizer
loads directly into the BabyLM evaluation harness via `AutoTokenizer.from_pretrained`.

## Training data

**Dataset creation**
A dataset can be constructed from books through the following command

```
python examples/gutenberg_download/download.py --target-mb 200 --output data/corpus.txt
```

Use `--delay` to adjust the pause between requests (default 1 s).
Quality polishing:
The program drops similar texts and documents with a high percentage of non alphabetic letters.
There is also a preprocess to remove repeated characters such as `\t` and `\n`, and cleaning up HTML tags.

**BabyLM 2026 Strict-Small**
The official 10M-token [BabyLM 2026 Strict-Small](https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict-Small)
corpus can be downloaded and concatenated into a single training file with

```
python examples/babylm_download/download.py --output data/babylm_strict_small.txt
```

The upstream text is already cleaned and detoxified, so no extra preprocessing is applied.

## Training

Example with a custom corpus:

```
python examples/generative_lm/train.py \
    --data data/corpus.txt \
    --output model.pt \
    --save-tokenizer tokenizer.json \
    --steps 10000 \
    --d-model 512 \
    --nhead 8 \
    --num-layers 6 \
    --vocab-size 4000
```

Resume training from a checkpoint:

```
python examples/generative_lm/train.py --resume model.pt --steps 5000
```

## Generation

```
python examples/generative_lm/generate.py \
    --model model.pt \
    --prompt "ROMEO:" \
    --steps 500 \
    --temperature 0.8
```
