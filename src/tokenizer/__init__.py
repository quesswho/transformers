import os as _os

# Python 3.8+ requires DLL search directories to be explicitly registered.
# libwinpthread-1.dll is shipped alongside the extension in this package dir.
_os.add_dll_directory(_os.path.dirname(__file__))
from ._sentencepiece_bpe import SentencePieceBPE

__all__ = ["SentencePieceBPE"]
