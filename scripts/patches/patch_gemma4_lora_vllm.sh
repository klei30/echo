#!/bin/bash
set -euo pipefail

TARGET="/home/klei/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/gemma4_mm.py"
BACKUP="${TARGET}.echo_backup_$(date +%Y%m%d_%H%M%S)"

cp "$TARGET" "$BACKUP"

/home/klei/vllm-env/bin/python - <<'PY'
from pathlib import Path

path = Path("/home/klei/vllm-env/lib/python3.12/site-packages/vllm/model_executor/models/gemma4_mm.py")
text = path.read_text()

text = text.replace(
    "    SupportsEagle3,\n    SupportsMultiModal,\n",
    "    SupportsEagle3,\n    SupportsLoRA,\n    SupportsMultiModal,\n",
)
text = text.replace(
    "    SupportsPP,\n    SupportsEagle3,\n):\n",
    "    SupportsPP,\n    SupportsEagle3,\n    SupportsLoRA,\n):\n",
)

path.write_text(text)
PY

/home/klei/vllm-env/bin/python -m py_compile "$TARGET"
grep -n -e "SupportsLoRA" -e "class Gemma4ForConditionalGeneration" "$TARGET"
echo "backup=$BACKUP"
