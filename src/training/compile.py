"""torch.compile environment workarounds."""

import sys


def _long_paths_enabled() -> bool:
    """Whether Windows long-path support is on in the registry (lets paths
    exceed the legacy 260-char MAX_PATH). Assumed off if it can't be read."""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        )
        return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def shorten_inductor_kernel_names() -> None:
    """Keep Inductor's generated Triton cache paths under the Windows MAX_PATH.

    By default Inductor gives each Triton kernel a *descriptive* name that
    concatenates every fused op (e.g. ``triton_per_fused__fused_rms_norm__
    fused_rms_norm_backward__to_copy_add_embedding_dense_backward_...``). The
    backward kernels of this model produce names long enough that the full
    ``.ttir`` cache path exceeds the legacy 260-char MAX_PATH; Triton then can't
    reopen the file mid-compile and compilation dies with a FileNotFoundError.
    Switching to short generic names (``triton_per_fused_0`` ...) keeps the path
    short. The only cost is less descriptive kernel names in profiles.

    No-op off Windows, or when long-path support is enabled, where descriptive
    names cause no trouble and are worth keeping.
    """
    if sys.platform != "win32" or _long_paths_enabled():
        return
    import torch._inductor.config as inductor_config
    inductor_config.triton.descriptive_names = False
