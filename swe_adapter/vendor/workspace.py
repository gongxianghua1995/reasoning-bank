"""SWE workspace primitives adapted from GPTSwarm; no graph or role scheduling."""
import asyncio
import hashlib
import json
import os
import re
import shlex
import sys
import time
import uuid
from pathlib import Path
from .mini_benchmark import is_pro, image_name, task_text
from .mini_repository import isolate_history_command


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def permission_only_diff(raw):
    """Accept only regular-file executable-bit changes with identical blobs.

    Input is git diff --raw --no-abbrev -z, without rename detection.
    Symlinks, content changes, additions, removals and malformed records fail.
    """
    records = raw.split('\0')
    if not raw or records[-1] != '' or len(records) % 2 != 1:
        return False
    for index in range(0, len(records)-1, 2):
        fields = records[index].split()
        if (len(fields) != 5 or not records[index+1] or fields[4] != 'M'
                or fields[0] not in {':100644', ':100755'}
                or fields[1] not in {'100644', '100755'}
                or fields[0][1:] == fields[1] or fields[2] != fields[3]):
            return False
    return True


def is_test_path(path):
    parts = Path(path).parts
    return (any(p in {'test', 'tests', '__tests__'} for p in parts)
            or Path(path).name.startswith('test_') or path.endswith('_test.go')
            or bool(re.search(r'\.(?:test|spec)\.[cm]?[jt]sx?$', path)))


async def command(argv, *, seconds=120, input_text=None, check=True):
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_text.encode() if input_text is not None else None), seconds)
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    result = {'returncode': proc.returncode, 'output': stdout.decode(errors='replace'),
              'stderr': stderr.decode(errors='replace')}
    if check and proc.returncode:
        raise RuntimeError(f'Command {argv[0]} exited {proc.returncode}: {result["stderr"][-1500:]}')
    return result


def merge_messages(inputs):
    messages = {}
    for value in inputs:
        for message in value.get('messages', []):
            messages[message['id']] = message
    return list(messages.values())


def review_verdict(report, exit_status):
    if exit_status != 'Submitted':
        return 'unverified'
    # A whole-line protocol; prose such as "not APPROVE" cannot approve a patch.
    matches = re.findall(r'^VERDICT: (APPROVE|REQUEST_CHANGES|UNVERIFIED)\s*$', report, re.M)
    return matches[0].lower() if len(matches) == 1 else 'unverified'


def interrupted_handoff(role, status, observations, partial_report=''):
    """Preserve concrete work even when a role cannot formally submit in time."""
    report = f'{role} exited with {status}; this is an incomplete handoff, not approval.\n'
    if partial_report.strip():
        report += '\nPartial report recovered from workspace:\n' + partial_report[:16000]
    report += '\nRecent tool observations (may include unsuccessful checks):\n'
    report += json.dumps(observations[-8:], ensure_ascii=False)
    return report


class Workspace:
    def __init__(self, record, config, out, mini_python=sys.executable):
        self.record = record
        self.config = config
        self.out = Path(out).resolve()
        self.mini_python = mini_python
        self.started = time.monotonic()
        self.deadline = self.started + config['task_seconds']
        self.containers = set()
        self.initial_untracked = {}
        self.main = None
        self.base = record['base_commit']
        self.cwd = '/app' if is_pro(record) else '/testbed'
        self.shell_env = [] if is_pro(record) else ['BASH_ENV=/root/.bashrc']
        if record.get('repo') == 'ansible/ansible':
            self.shell_env.append('PYTHONPATH=/app/lib')

    def event(self, event, **values):
        with (self.out / 'runtime.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, elapsed=time.monotonic()-self.started,
                                         **values), ensure_ascii=False) + '\n')

    async def shell(self, container, script, *, seconds=120, input_text=None, check=True):
        argv = ['docker', 'exec', '-i', '-w', self.cwd,
                *[part for value in self.shell_env for part in ['-e', value]],
                container, 'timeout', '-k', '2', str(seconds), 'bash', '-o', 'pipefail', '-c', script]
        return await command(argv, seconds=seconds+5, input_text=input_text, check=check)

    async def start_workspace(self, purpose, patch=''):
        name = 'reasoningbank_swe_' + purpose + '_' + uuid.uuid4().hex[:12]
        image = image_name(self.record)
        self.containers.add(name)
        await command(['docker', 'run', '-d', '--name', name, '--network', 'none',
                       '--label', 'reasoningbank.swe.run=' + self.out.name,
                       '-w', self.cwd, '--entrypoint', '/bin/bash', image,
                       '-c', 'exec tail -f /dev/null'])
        inspection = json.loads((await command(['docker', 'inspect', name]))['output'])[0]
        if (inspection['HostConfig']['NetworkMode'] != 'none'
                or any(n != 'none' for n in inspection['NetworkSettings']['Networks'])):
            raise RuntimeError('Generation container is not network isolated')
        head = (await self.shell(name, 'git rev-parse HEAD'))['output'].strip()
        if head != self.base:
            # Image builders may commit setup edits (e.g. Sphinx's tox.ini) or
            # chmod changes. Restore the exact dataset base BEFORE any agent
            # can inspect code; the image HEAD is not the generation baseline.
            trees = (await self.shell(name, 'git rev-parse HEAD^{tree} ' +
                                     shlex.quote(self.base + '^{tree}')))['output'].splitlines()
            if len(trees) != 2:
                raise RuntimeError('Image source tree differs from task base commit')
            if trees[0] != trees[1]:
                raw = (await self.shell(name, 'git diff --raw --no-abbrev --no-renames -z ' +
                                        shlex.quote(self.base) + ' HEAD'))['output']
                self.event('image_permission_difference' if permission_only_diff(raw)
                           else 'image_baseline_difference', container=name, image=image,
                           files=len(raw.split('\0'))//2,
                           policy='restore exact dataset base before agent access')
        if is_pro(self.record):
            # Pro installers update dependency manifests without changing
            # installed dependencies. Restore only these known setup artifacts;
            # unexpected tracked source edits remain a preparation error.
            dirty = (await self.shell(name, 'git diff HEAD --ignore-submodules=untracked --name-only -z'))['output']
            setup_paths = set(filter(None, dirty.split('\0')))
            allowed = {'yarn.lock', 'package-lock.json', 'requirements.txt'}
            if self.record['repo'] == 'flipt-io/flipt':
                allowed.add('go.work.sum')  # populated by cached module installation
            if self.record['repo'] == 'internetarchive/openlibrary':
                # Image's Selenium/headless/PyYAML compatibility edits. These
                # are test setup, not candidate code; restore exact task base.
                allowed.add('tests/integration/__init__.py')
            if self.record['repo'] == 'protonmail/webclients':
                allowed.update({'applications/drive/public/assets/sandbox.js',
                                'applications/mail/public/assets/sandbox.js'})
            if setup_paths - allowed:
                raise RuntimeError('Unexpected Pro image source changes: ' + repr(sorted(setup_paths - allowed)))
            if setup_paths:
                await self.shell(name, 'git checkout HEAD -- ' + ' '.join(shlex.quote(p) for p in sorted(setup_paths)))
                self.event('image_setup_restored', files=sorted(setup_paths), container=name)
            # A preinstalled submodule may contain an untracked import symlink.
            # Preserve it and its dependency tree, but remove all future history.
            await self.shell(name, 'git config diff.ignoreSubmodules untracked')
        dirty = (await self.shell(name, 'git diff HEAD --name-only'))['output']
        if dirty.strip():
            raise RuntimeError('Image contains tracked modifications before generation')
        await self.shell(name, 'git checkout --detach ' + shlex.quote(self.base))
        restored = (await self.shell(name, 'git rev-parse HEAD'))['output'].strip()
        if restored != self.base:
            raise RuntimeError('Failed to restore exact task base commit')
        await self.shell(name, 'git diff --exit-code HEAD -- && git diff --cached --exit-code HEAD --')
        # Clone only the current commit over the local transport. Remove the
        # original object store, refs and reflogs from this disposable container.
        script = isolate_history_command(self.base) if is_pro(self.record) else '''set -eu
swarm_git_tmp=$(mktemp -d /tmp/swarm-git.XXXXXXXX)
git clone --quiet --no-local --depth 1 --no-checkout "file://$PWD" "$swarm_git_tmp"
rm -rf .git
mv "$swarm_git_tmp/.git" .git
rmdir "$swarm_git_tmp"
git reset --mixed HEAD >/dev/null
git remote remove origin
test "$(git rev-list --all --count)" = 1
'''
        await self.shell(name, script)
        if is_pro(self.record):
            await self.shell(name, 'git config diff.ignoreSubmodules untracked')
        shallow = (await self.shell(name, 'git rev-parse --is-shallow-repository'))['output'].strip()
        if shallow != 'true':
            raise RuntimeError('Expected a base-only shallow repository')
        initial = (await self.shell(name, 'git ls-files --others --exclude-standard -z'))['output']
        self.initial_untracked[name] = set(filter(None, initial.split('\0')))
        if patch:
            await self.shell(name, 'git apply --binary -', input_text=patch)
        self.event('workspace_started', purpose=purpose, container=name, image=image,
                   network=inspection['HostConfig']['NetworkMode'], image_head=head, base_commit=self.base,
                   shallow=True, patch_sha256=digest(patch))
        return name

    async def prepare(self):
        self.main = await self.start_workspace('engineer')

    async def export(self, container=None, *, include_tests=False):
        container = container or self.main
        if not container:
            return {'patch': '', 'sha256': digest(''), 'excluded_paths': []}
        changed = (await self.shell(container, 'git diff HEAD --name-only -z'))['output']
        untracked = (await self.shell(container, 'git ls-files --others --exclude-standard -z'))['output']
        paths = set(filter(None, changed.split('\0')))
        paths.update(set(filter(None, untracked.split('\0'))) - self.initial_untracked[container])
        excluded = sorted(p for p in paths if not include_tests and is_test_path(p))
        paths -= set(excluded)
        if not paths:
            return {'patch': '', 'sha256': digest(''), 'excluded_paths': excluded}
        # A separate index includes unstaged, staged, added and deleted files
        # without changing the agent's index or relying on model diff text.
        script = '''set -eu
swarm_index=$(mktemp /tmp/swarm-index.XXXXXXXX)
rm "$swarm_index"
export GIT_INDEX_FILE="$swarm_index"
trap 'rm -f "$GIT_INDEX_FILE"' EXIT
git read-tree HEAD
git add -A -- ''' + ' '.join(shlex.quote(':(literal)' + p) for p in sorted(paths)) + '''
git diff --cached --binary HEAD
'''
        patch = (await self.shell(container, script))['output']
        return {'patch': patch, 'sha256': digest(patch), 'excluded_paths': excluded}

    async def stop_workspace(self, name):
        await command(['docker', 'rm', '-f', name], check=False)
        self.containers.discard(name)

    async def close(self):
        for name in list(self.containers):
            await self.stop_workspace(name)

