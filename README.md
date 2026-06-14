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

## GPT-BERT (joint causal + masked training)

`examples/gpt_bert/train.py` trains a single transformer as both a GPT and a
BERT, following [Charpentier & Samuel (2024)](https://arxiv.org/abs/2410.24159).
Each step draws one of two objectives:

- **causal** next-token prediction (a causal-masked window), and
- **masked-next-token prediction (MNTP)**: a bidirectional, mask-corrupted
  window whose masked-LM labels are shifted by one, so a masked token is
  predicted from the *previous* position's hidden state — the same output offset
  the causal objective uses. One shared output head therefore serves both.

```
python examples/gpt_bert/train.py \
    --data data/babylm_strict_small.txt \
    --output models/gptbert.pt \
    --save-tokenizer data/vocab/gptbert10k.json \
    --vocab-size 10000 \
    --steps 10000 \
    --mlm-ratio 0.5 \
    --mlm-prob 0.15
```

`--mlm-ratio` is the fraction of steps using the MNTP objective; `--mlm-prob` is
the per-token masking probability within an MNTP batch. The model is
architecturally identical to the decoder-only model, so its checkpoints load
directly into `generate.py`, the HF export, and the BabyLM eval (all of which
score causally). The tokenizer gains a `<mask>` token; train a fresh one (or
reuse a tokenizer built with the current code) so MNTP has a mask id to use.

## Generation

```
python examples/generative_lm/generate.py \
    --model model.pt \
    --prompt "ROMEO:" \
    --steps 500 \
    --temperature 0.8
```
