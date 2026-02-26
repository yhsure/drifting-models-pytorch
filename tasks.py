"""Invoke task definitions."""

from __future__ import annotations

import shlex

from invoke import Context, task


@task
def format(c: Context) -> None:
    """Format code with ruff."""
    c.run("uv run ruff format .")


@task
def lint(c: Context) -> None:
    """Lint and autofix code with ruff."""
    c.run("uv run ruff check . --fix")


@task
def typecheck(c: Context) -> None:
    """Run static type checks."""
    c.run("uv run ty check")


@task
def test(c: Context) -> None:
    """Run tests."""
    c.run("uv run pytest tests/")


@task
def precommit(c: Context) -> None:
    """Run all pre-commit hooks."""
    c.run("uv run pre-commit run --all-files")


@task(help={"m": "Commit message."})
def commit(c: Context, m: str) -> None:
    """Run pre-commit with autofixes and create a commit if checks pass."""
    first = c.run("uv run pre-commit run", warn=True)
    if first.exited != 0:
        c.run("git add -A")
        c.run("uv run pre-commit run")
    c.run("git add -A")
    c.run(f"git commit -m {shlex.quote(m)}")
