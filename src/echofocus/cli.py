"""CLI entrypoint for EchoFocus."""

import fire

from .echofocus import EchoFocus


def main():
    """Run the Fire-based EchoFocus CLI."""
    fire.Fire(EchoFocus)
