#!/usr/bin/env bash
#MISE description="deletes .ruff_cache/, __pycache__ and .venv/"
#MISE quiet=true

rm -rf __pycache__
rm -rf .ruff_cache
rm -rf .venv
