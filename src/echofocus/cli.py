"""CLI entrypoint for EchoFocus."""

import fire

from .app import EchoFocus


def main():
    """Run the Fire-based EchoFocus CLI."""
    fire.Fire(EchoFocus)
