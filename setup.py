import os
import shutil
import subprocess
import sys
import sysconfig

from pybind11.setup_helpers import Pybind11Extension, build_ext as Pybind11BuildExt
from setuptools import setup

_have_msvc = sys.platform == "win32" and bool(shutil.which("cl"))

if sys.platform == "win32":
    compile_args = ["/O2", "/utf-8"] if _have_msvc else ["-O2"]
else:
    compile_args = ["-O2"]


class build_ext(Pybind11BuildExt):
    """Falls back to a direct g++ call on Windows when MSVC is absent."""

    _gcc_built: set = set()

    def build_extension(self, ext):
        if sys.platform == "win32" and not _have_msvc:
            self._gcc_build(ext)
            self._gcc_built.add(ext.name)
        else:
            super().build_extension(ext)

    def copy_extensions_to_source(self):
        # _gcc_build writes directly to the source tree; skip the copy for those.
        original = self.extensions
        self.extensions = [e for e in self.extensions if e.name not in self._gcc_built]
        super().copy_extensions_to_source()
        self.extensions = original

    def _gcc_build(self, ext):
        import pybind11

        py_inc  = sysconfig.get_config_var("INCLUDEPY")
        py_libs = os.path.join(sys.base_prefix, "libs")
        pb_inc  = pybind11.get_include()
        suffix  = sysconfig.get_config_var("EXT_SUFFIX")  # e.g. .cp312-win_amd64.pyd
        pyver   = f"python{sys.version_info.major}{sys.version_info.minor}"

        # Write directly to the source tree (the --inplace destination).
        out = os.path.join("src", "tokenizer", f"_sentencepiece_bpe{suffix}")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

        cmd = [
            "g++", "-shared",
            "-static-libgcc", "-static-libstdc++",
            f"-std=c++{ext.cxx_std}",
            "-O2",
            f"-I{py_inc}", f"-I{pb_inc}",
            *ext.sources,
            "-o", out,
            f"-L{py_libs}", f"-l{pyver}",
            "-Wl,--enable-auto-import",
        ]
        print("Building with GCC:", " ".join(cmd))
        subprocess.run(cmd, check=True)


ext_modules = [
    Pybind11Extension(
        "src.tokenizer._sentencepiece_bpe",
        sources=["src/tokenizer/sentencepiece_bpe.cpp"],
        extra_compile_args=compile_args,
        cxx_std=17,
    ),
]

setup(
    name="transformers-tokenizer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
