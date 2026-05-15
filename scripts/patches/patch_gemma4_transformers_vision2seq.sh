#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/klei/gemma4-transformers/transformers"
AUTO_FILE="$ROOT/models/auto/modeling_auto.py"

if [ ! -f "$AUTO_FILE" ]; then
  echo "Gemma4 transformers auto model file not found at $AUTO_FILE" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
cp "$AUTO_FILE" "$AUTO_FILE.echo_vision2seq_backup_$STAMP"

python3 - <<'PY'
from pathlib import Path

auto_file = Path("/home/klei/gemma4-transformers/transformers/models/auto/modeling_auto.py")
text = auto_file.read_text()

alias = "\n\nAutoModelForVision2Seq = AutoModelForImageTextToText\n"
if "AutoModelForVision2Seq = AutoModelForImageTextToText" not in text:
    marker = "AutoModelForImageTextToText = auto_class_update(AutoModelForImageTextToText, head_doc=\"image-text-to-text modeling\")"
    if marker not in text:
        raise SystemExit("Could not find AutoModelForImageTextToText marker")
    text = text.replace(marker, marker + alias, 1)

if '"AutoModelForVision2Seq",' not in text:
    text = text.replace('"AutoModelForImageTextToText",\n', '"AutoModelForImageTextToText",\n    "AutoModelForVision2Seq",\n', 1)

auto_file.write_text(text)
PY

echo "Patched AutoModelForVision2Seq compatibility into $AUTO_FILE"
