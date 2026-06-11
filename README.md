# Transformers

Pytorch implementation of decoder transformers

Includes a next token predictor in examples/generative_lm

## Build

The following is subject to change:

The BPE tokenizer has a C++ core (pybind11 extension) that must be compiled before use.

**Requirements:** [Visual Studio Build Tools 2026](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload, CMake 3.21+, pybind11 (`pip install pybind11>=2.12`).

```powershell
cmake -B build -G "Visual Studio 18 2026" -A x64 -DPython3_EXECUTABLE=.venv\Scripts\python.exe
cmake --build build --config Release
```

Replace `.venv\Scripts\python.exe` with the path to your Python 3.12 interpreter if not using a local venv. The compiled `.pyd` is placed in `src/tokenizer/` automatically.

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
