#!/bin/sh
set -e
cd "$(dirname "$0")/.."
cp _merge/README.md _merge/BASELINE.md _merge/.gitignore .
git add README.md BASELINE.md .gitignore
git status --short | grep '^UU' && { echo "unresolved conflicts left"; exit 1; }
rm -r _merge
git commit --no-edit
