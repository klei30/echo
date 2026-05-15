#!/usr/bin/env bash
set -euo pipefail

TARGET="/home/klei/vllm-env/lib/python3.12/site-packages/llamafactory/data/template.py"
MARKER='name="gemma4"'

if grep -q "$MARKER" "$TARGET"; then
  echo "LlamaFactory Gemma4 template already patched"
  exit 0
fi

python3 - <<'PY'
from pathlib import Path

path = Path("/home/klei/vllm-env/lib/python3.12/site-packages/llamafactory/data/template.py")
text = path.read_text()

anchor = '''# copied from gemma template
register_template(
    name="gemma3",
'''

insert = '''# Gemma 4 uses the tokenizer-native turn tokens, not Gemma 1/2
# <start_of_turn>/<end_of_turn>. Using the older template can add an
# out-of-range token id during SFT.
register_template(
    name="gemma4",
    format_user=StringFormatter(slots=["<|turn>user\\n{{content}}<turn|>\\n<|turn>model\\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<turn|>\\n"]),
    format_system=StringFormatter(slots=["{{content}}\\n\\n"]),
    format_observation=StringFormatter(slots=["<|turn>tool\\n{{content}}<turn|>\\n<|turn>model\\n"]),
    format_prefix=EmptyFormatter(slots=[{"bos_token"}]),
    stop_words=["<turn|>"],
    efficient_eos=True,
    template_class=Llama2Template,
)


'''

if anchor not in text:
    raise SystemExit("Could not find Gemma3 template anchor")

path.write_text(text.replace(anchor, insert + anchor))
print("Patched LlamaFactory Gemma4 template")
PY
