#!/usr/bin/env bash
set -euo pipefail

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to publish from a dirty working tree." >&2
    exit 1
fi

release_dir="$(mktemp -d)"
trap 'rm -rf "$release_dir"' EXIT

python -m build --outdir "$release_dir"
python -m twine check "$release_dir"/*
python -m twine upload "$release_dir"/*
