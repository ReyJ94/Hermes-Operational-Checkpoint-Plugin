try:
    from .operational_checkpoint import (
        OperationalCheckpointCompressor,
        install_plugin_sidecar,
        register,
    )
except ImportError:
    from operational_checkpoint import (
        OperationalCheckpointCompressor,
        install_plugin_sidecar,
        register,
    )

__all__: list[str] = [
    "OperationalCheckpointCompressor",
    "install_plugin_sidecar",
    "register",
]
