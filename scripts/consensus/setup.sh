#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
project_storage="${1:-${MEANFLOW_STORAGE:-$root/work}}"
mkdir -p -- "$project_storage"
uv venv --python 3.12 --allow-existing "$project_storage/meanflow-env"
uv pip install --python "$project_storage/meanflow-env/bin/python" -r "$root/scripts/consensus/environment-inference.lock.txt"
uv venv --python 3.12 --allow-existing "$project_storage/meanflow-features-env"
uv pip install --python "$project_storage/meanflow-features-env/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r "$root/scripts/consensus/environment-features.lock.txt"
printf 'Inference interpreter: %s\nFeature interpreter: %s\n' "$project_storage/meanflow-env/bin/python" "$project_storage/meanflow-features-env/bin/python"
