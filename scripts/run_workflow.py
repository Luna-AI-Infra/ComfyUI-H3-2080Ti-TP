"""Check a graph against a live server; optionally submit exactly once and wait."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8188')
    ap.add_argument('--workflow', required=True, type=Path)
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--timeout', type=float, default=3600)
    args = ap.parse_args()
    base = args.url.rstrip('/')

    def api(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=30) as f:
                return json.load(f)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")}') from None

    graph = json.loads(args.workflow.read_text(encoding='utf-8'))
    if 'nodes' in graph:
        ap.error('This is editor JSON; select a file from workflows/api')
    info = api('/object_info')
    errors = []
    for node_id, node in graph.items():
        cls = node.get('class_type')
        if cls not in info:
            errors.append(f'{node_id}: missing node {cls}')
            continue
        supplied = node.get('inputs', {})
        inputs = info[cls].get('input', {})
        for name in inputs.get('required', {}):
            if name not in supplied:
                errors.append(f'{node_id}/{cls}: missing required input {name}')
        for name, schema in {**inputs.get('required', {}), **inputs.get('optional', {})}.items():
            value = supplied.get(name)
            linked = isinstance(value, list) and len(value) == 2 and str(value[0]) in graph and isinstance(value[1], int)
            if name in supplied and not linked and isinstance(schema[0], list) and value not in schema[0]:
                errors.append(f'{node_id}/{cls}: enum/model choice not available: {name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Live registration/input check passed; complete graph validation happens on submission.')
    if not args.submit:
        return
    result = api('/prompt', dict(prompt=graph, client_id='h3-tp-' + str(uuid.uuid4())))
    prompt_id = result['prompt_id']
    print('Submitted once:', prompt_id, flush=True)
    start = time.monotonic()
    while time.monotonic() - start < args.timeout:
        history = api('/history/' + prompt_id)
        if prompt_id in history:
            record = history[prompt_id]
            if record['status']['status_str'] != 'success':
                raise SystemExit('Execution failed; no retry. Inspect history and stop on fatal GPU errors.')
            stamps = {k: v['timestamp'] for k, v in record['status']['messages']
                      if k in ('execution_start', 'execution_success')}
            print(json.dumps(dict(prompt_id=prompt_id, request_seconds=(stamps['execution_success']-stamps['execution_start'])/1000,
                                  outputs=record['outputs']), indent=2))
            return
        time.sleep(1)
    raise SystemExit('Timed out waiting for history; request may still be queued/running. No duplicate submitted.')


if __name__ == '__main__':
    main()
