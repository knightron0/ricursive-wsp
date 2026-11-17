"""Let an interactive agent (for example a Lor session on another model) play one episode through a shell.

    python -m wsp.bridge serve <wsp args>     start the episode and serve it on RUN/bridge.sock (run in the background)
    python -m wsp.bridge call RUN < calls.json  one reply: a JSON list of {"name", "args"}; prints the results

A call() is one "reply" for the turn count, exactly as in the API loop: calls run in order, a submit() ends the
episode, a reply whose calls were all format rejections does not count. view() images are written to
RUN/views/*.png and the result lists the paths (the agent opens them with its image reader). When submit() is
accepted the server scores the diff on held-out val (and test with --evaluate-test), writes result.json and
audit.json, and exits.

This is not isolated like the API loop: the agent has a shell on the same machine and nothing here stops it from
reading files outside the workspace. It is meant for agents that are told not to.
"""
import json
import os
from pathlib import Path
import socket
import socketserver
import sys

SOCK = 'bridge.sock'


def client(run, calls):
    s = socket.socket(socket.AF_UNIX)
    s.connect(str(Path(run) / SOCK))
    s.sendall(json.dumps(calls).encode() + b'\n')
    buf = b''
    while not buf.endswith(b'\n'):
        chunk = s.recv(1 << 16)
        if not chunk:
            break
        buf += chunk
    return json.loads(buf)['text']


def serve(argv):
    from functools import partial
    import torch
    from wsp import data as datasets
    from wsp.__main__ import main, save, summarize
    from wsp.agent import turn_note
    from wsp.episode import Episode, Visual, add_note, audit
    from wsp.evaluation import evaluate
    from wsp.tasks import TASKS, apply_diff, start_weights

    main(argv + ['--dry-run'])  # writes config.json, subset_indices.json and prompt.json
    out = Path(next(argv[i + 1] for i, x in enumerate(argv) if x == '--output'))
    a = json.loads((out / 'config.json').read_text())
    task = TASKS[a['task']]
    data_dir = Path(a['data_dir'])
    start = start_weights(task, torch.load(a['source'], weights_only=True) if task.needs_source else None)
    classes = datasets.class_names(task.dataset, data_dir)
    images, labels = datasets.load_split(task.dataset, 'train', data_dir)
    subset = torch.tensor(json.loads((out / 'subset_indices.json').read_text())['indices'], dtype=torch.long)
    images, labels = images[subset].clone(), labels[subset].clone()
    log = []
    episode = Episode(a['mode'], start, images, labels, classes, out / 'workspace',
                      partial(apply_diff, max_numbers=a['max_numbers']), a['device'], a['tool_timeout'], log,
                      a['visual'])
    state = {'turn': 0, 'retries': a['format_retries'], 'views': 0, 'image_blocks': 0}
    (out / 'views').mkdir(exist_ok=True)

    def render(name, text, error):
        if isinstance(text, Visual):
            lines = []
            for p in text.parts:
                if isinstance(p, str):
                    lines.append(p)
                else:
                    state['views'] += 1
                    state['image_blocks'] += 1
                    path = out / 'views' / f'view_{state["views"]:03d}.png'
                    path.write_bytes(p)
                    lines.append(f'[image saved: {path}]')
            text = '\n'.join(lines)
        return f'--- {name}{" (ERROR)" if error else ""} ---\n{text}'

    def finish():
        problems = audit(a['mode'], log, ['image'] * state['image_blocks'], a['visual'])
        save(out / 'calls.json', log)
        save(out / 'audit.json', {'clean': not problems, 'problems': problems})
        if episode.submission is None:
            save(out / 'result.json', {'error': 'no valid submission'})
            return 'No valid submission: the episode is over with no score.'
        torch.save(episode.submission, out / 'weights.pt')
        result = {'calls': len(log)}
        for split in ['val'] + (['test'] if a['evaluate_test'] else []):
            held_out = datasets.load_split(task.dataset, split, data_dir)
            result[split] = {n: summarize(evaluate(w, held_out, classes, a['device']))
                             for n, w in (('start', start), ('submitted', episode.submission))}
        save(out / 'result.json', result)
        return f'Submission accepted. Held-out result: {json.dumps(result)}'

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            calls = json.loads(self.rfile.readline())
            max_turns = a['max_turns']
            if episode.submission is not None or state['turn'] >= max_turns:
                text, done = 'The episode is over.', True
            else:
                first, parts = len(log), []
                for c in calls:
                    t, err = episode.call(c['name'], c.get('args') or {})
                    parts.append(render(c['name'], t, err))
                    if episode.submission is not None:
                        break
                entries = log[first:]
                counted = not (state['retries'] > 0 and entries and all(e['status'] == 'format' for e in entries))
                if counted:
                    state['turn'] += 1
                else:
                    state['retries'] -= 1
                text = '\n'.join(parts)
                if episode.submission is None:
                    text += '\n\n' + turn_note(episode, state['turn'], max_turns, counted, state['retries'])
                done = episode.submission is not None or state['turn'] >= max_turns
                if done:
                    text += '\n\n' + finish()
            self.wfile.write(json.dumps({'text': text}).encode() + b'\n')
            if done:
                self.server.done = True

    sock = out / SOCK
    if sock.exists():
        sock.unlink()
    with socketserver.UnixStreamServer(str(sock), Handler) as server:
        server.done = False
        print(f'bridge ready: {out}', flush=True)
        while not server.done:
            server.handle_request()
    print(f'episode finished; see {out}/result.json', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'serve':
        serve(sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == 'call':
        print(client(sys.argv[2], json.load(sys.stdin)))
    else:
        sys.exit(__doc__)
