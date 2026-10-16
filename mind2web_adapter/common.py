import hashlib
import json
from pathlib import Path
import threading
import time


def load(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def event(out, name, **values):
    with (Path(out) / 'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name, time=time.time(), **values), ensure_ascii=False) + '\n')


class Progress:
    def __init__(self, out, **values):
        self.out = Path(out)
        self.state = dict(status='running', started=time.time(), **values)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.pulse, daemon=True)
        self.update()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.state.update(values)
            self.state['elapsed_seconds'] = round(time.time() - self.state['started'], 2)
            self.state['updated'] = time.time()
            write_json(self.out / 'status.json', self.state)

    def pulse(self):
        while not self.stop.wait(15):
            self.update()
            print(json.dumps(self.state, ensure_ascii=False), flush=True)

    def close(self, **values):
        self.stop.set()
        self.thread.join()
        self.update(**values)
