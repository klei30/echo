import json
import uuid
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).parent / "echo_local_smoke_test.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    code_cell(
        "import json, os, time, uuid\n"
        "from pathlib import Path\n"
        "import requests\n"
        "\n"
        "BASE_URL = os.getenv('ECHO_BASE_URL', 'http://127.0.0.1:8002').rstrip('/')\n"
        "VLLM_BASE_URL = os.getenv('GEMMA4_VLLM_BASE_URL', 'http://127.0.0.1:8003/v1').rstrip('/')\n"
        "MODEL = os.getenv('GEMMA4_BASE_MODEL', 'gemma4_e2b')\n"
        "DEMO_SEED_TOKEN = os.getenv('ECHO_DEMO_SEED_TOKEN', 'kaggle-demo-seed')\n"
        "TOKEN = os.getenv('ECHO_TOKEN', '')\n"
        "USER_ID = ''\n"
        "\n"
        "def show(title, value):\n"
        "    print('\\n' + '=' * 80)\n"
        "    print(title)\n"
        "    print('=' * 80)\n"
        "    if isinstance(value, (dict, list)):\n"
        "        print(json.dumps(value, indent=2)[:5000])\n"
        "    else:\n"
        "        print(value)\n"
        "\n"
        "def headers():\n"
        "    return {'Authorization': f'Bearer {TOKEN}'} if TOKEN else {}\n"
        "\n"
        "def echo(method, path, payload=None, params=None, timeout=180):\n"
        "    print(f'Echo {method} {path} ...', flush=True)\n"
        "    r = requests.request(method, BASE_URL + path, headers=headers(), json=payload, params=params, timeout=timeout)\n"
        "    print(f'-> HTTP {r.status_code}', flush=True)\n"
        "    r.raise_for_status()\n"
        "    return r.json() if r.text else {}\n"
        "\n"
        "print('Config loaded')\n"
        "print('BASE_URL =', BASE_URL)\n"
        "print('VLLM_BASE_URL =', VLLM_BASE_URL)\n"
        "print('MODEL =', MODEL)\n"
    ),
    code_cell(
        "print('Checking Echo and vLLM health...', flush=True)\n"
        "echo_health = requests.get(BASE_URL + '/health', timeout=10).json()\n"
        "models = requests.get(VLLM_BASE_URL + '/models', timeout=10).json()\n"
        "show('Echo health', echo_health)\n"
        "show('vLLM models', models)\n"
        "assert any(m.get('id') == MODEL for m in models.get('data', [])), f'{MODEL} not found in vLLM models'\n"
        "print('Health checks passed')\n"
    ),
    code_cell(
        "global TOKEN, USER_ID\n"
        "print('Seeding public-safe demo user...', flush=True)\n"
        "seed = requests.post(\n"
        "    BASE_URL + '/v1/demo/seed',\n"
        "    headers={'x-echo-demo-token': DEMO_SEED_TOKEN},\n"
        "    json={'scenario': 'proof_camera_maya', 'reset': False, 'stable': False},\n"
        "    timeout=180,\n"
        ")\n"
        "print('seed HTTP', seed.status_code, flush=True)\n"
        "seed.raise_for_status()\n"
        "seed_json = seed.json()\n"
        "TOKEN = seed_json['token']\n"
        "USER_ID = seed_json['user']['id']\n"
        "show('Seed result', {'user_id': USER_ID, 'result': seed_json.get('result')})\n"
        "show('Authenticated user', echo('GET', '/auth/me'))\n"
    ),
    code_cell(
        "nonce = uuid.uuid4().hex[:10]\n"
        "print('Calling direct vLLM with nonce', nonce, flush=True)\n"
        "direct = requests.post(\n"
        "    VLLM_BASE_URL + '/chat/completions',\n"
        "    json={\n"
        "        'model': MODEL,\n"
        "        'messages': [{'role': 'user', 'content': f'Reply in one sentence and include nonce {nonce}.'}],\n"
        "        'temperature': 0.2,\n"
        "        'max_tokens': 80,\n"
        "    },\n"
        "    timeout=180,\n"
        ")\n"
        "direct.raise_for_status()\n"
        "direct_json = direct.json()\n"
        "text = direct_json['choices'][0]['message']['content']\n"
        "show('Direct live Gemma response', {'id': direct_json.get('id'), 'model': direct_json.get('model'), 'text': text})\n"
        "assert nonce in text, 'Gemma response did not include the nonce'\n"
    ),
    code_cell(
        "nonce = uuid.uuid4().hex[:10]\n"
        "chat = echo('POST', '/v1/chat/completions', {\n"
        "    'model': MODEL,\n"
        "    'messages': [{'role': 'user', 'content': f'This is a live Echo chat test. Mention nonce {nonce} and one proof step.'}],\n"
        "    'temperature': 0.3,\n"
        "    'max_tokens': 180,\n"
        "})\n"
        "content = chat['choices'][0]['message']['content']\n"
        "show('Echo live chat', {'id': chat.get('id'), 'model': chat.get('model'), 'content': content})\n"
        "print('Echo chat returned a live completion. Nonce requested:', nonce)\n"
    ),
    code_cell(
        "artifact = {\n"
        "    'artifact_type': 'mobile_camera_payload',\n"
        "    'scene': 'Garden sensor prototype beside field-test notes.',\n"
        "    'visible_text': [\n"
        "        'Garden sensor v2',\n"
        "        'Outdoor test stable for 40 minutes',\n"
        "        'Cost reduced from $18 to $11',\n"
        "        'Teacher says Maya explains electronics clearly',\n"
        "    ],\n"
        "    'user_caption': 'This worked outside without internet.',\n"
        "    'goal': 'Win a scholarship or apprenticeship by showing real technical proof.',\n"
        "    'opportunity_type': 'scholarship',\n"
        "}\n"
        "vision = echo('POST', '/v1/vision/analyze', artifact, timeout=240)\n"
        "show('Proof Camera / vision analyze', vision)\n"
    ),
    code_cell(
        "trace = echo('GET', '/v1/training/pipeline-trace', params={'lane': 'gemma4_e2b', 'prepare': '1', 'write': '1'}, timeout=300)\n"
        "show('Shadow Clone pipeline trace', {\n"
        "    'ready': trace.get('ready'),\n"
        "    'prep': trace.get('prep'),\n"
        "    'raw_counts': trace.get('raw_counts'),\n"
        "    'datasets': {k: {'rows': v.get('rows'), 'ready': v.get('ready'), 'path': v.get('path')} for k, v in (trace.get('datasets') or {}).items()},\n"
        "})\n"
    ),
    code_cell(
        "print('Triggering bounded real demo training now. This may restart vLLM.', flush=True)\n"
        "evidence = echo('POST', '/v1/training/demo-loop', {'lane': 'gemma4_e2b', 'max_pairs': 8, 'max_steps': 8, 'min_pairs': 4}, timeout=1800)\n"
        "show('Bounded demo training evidence', {\n"
        "    'status': evidence.get('status'),\n"
        "    'profile': evidence.get('profile'),\n"
        "    'real_training': evidence.get('real_training'),\n"
        "    'bounds': evidence.get('bounds'),\n"
        "    'dataset': evidence.get('dataset'),\n"
        "    'runtime_steps': evidence.get('runtime_steps'),\n"
        "    'promotion': evidence.get('promotion'),\n"
        "    'before_after': evidence.get('before_after'),\n"
        "})\n"
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "cells": cells,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
