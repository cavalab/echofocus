"""EchoFocus package."""

__all__ = ["EchoFocus"]


def __getattr__(name):
    if name == "EchoFocus":
        from .echofocus import EchoFocus

        return EchoFocus
    raise AttributeError(name)
