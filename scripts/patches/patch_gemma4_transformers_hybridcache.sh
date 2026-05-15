#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/klei/gemma4-transformers/transformers"
CACHE_UTILS="$ROOT/cache_utils.py"
INIT_FILE="$ROOT/__init__.py"

if [ ! -f "$CACHE_UTILS" ] || [ ! -f "$INIT_FILE" ]; then
  echo "Gemma4 transformers override not found at $ROOT" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
cp "$CACHE_UTILS" "$CACHE_UTILS.echo_backup_$STAMP"
cp "$INIT_FILE" "$INIT_FILE.echo_backup_$STAMP"

python3 - <<'PY'
from pathlib import Path

root = Path("/home/klei/gemma4-transformers/transformers")
cache_utils = root / "cache_utils.py"
init_file = root / "__init__.py"

cache_text = cache_utils.read_text()
if "class HybridCache(StaticCache):" not in cache_text:
    marker = "\n\nclass QuantizedCache(Cache):"
    patch = '''

class HybridCache(StaticCache):
    """
    Compatibility alias for PEFT versions that still import HybridCache.
    The Gemma4 Transformers override already folds hybrid/sliding behavior into StaticCache.
    """
'''
    if marker not in cache_text:
        raise SystemExit("Could not find QuantizedCache marker in cache_utils.py")
    cache_text = cache_text.replace(marker, patch + marker, 1)
    cache_utils.write_text(cache_text)

init_text = init_file.read_text()
if '"HybridCache",' not in init_text:
    init_text = init_text.replace('"EncoderDecoderCache",\n', '"EncoderDecoderCache",\n        "HybridCache",\n', 1)
if "from .cache_utils import HybridCache as HybridCache" not in init_text:
    init_text = init_text.replace(
        "from .cache_utils import EncoderDecoderCache as EncoderDecoderCache\n",
        "from .cache_utils import EncoderDecoderCache as EncoderDecoderCache\n    from .cache_utils import HybridCache as HybridCache\n",
        1,
    )
init_file.write_text(init_text)
PY

echo "Patched HybridCache compatibility into $ROOT"
