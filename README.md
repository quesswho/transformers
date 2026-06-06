# Transformers

Transformer implementations in PyTorch for educational purposes

## Build

The BPE tokenizer has a C++ core that must be compiled before use:

```
python setup.py build_ext --inplace
```

## Training data

**Quick start — tinyshakespeare (automatic)**

Omit `--data` and the trainer downloads the dataset automatically. Good for smoke-testing.

**Larger corpus — Project Gutenberg**

```
python examples/gutenberg_download/download.py --target-mb 200 --output data/corpus.txt
```

The default target is 500 MB. Use `--delay` to adjust the pause between requests (default 1 s).

## Training

Minimal run (tinyshakespeare downloaded automatically):

```
python examples/generative_lm/train.py --output model.pt
```

Full example with a custom corpus:

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
