"""OpenAI-compatible transport adapted from the existing SWE transport.

Only this transport is reused; Mind2Web does not use the SWE agent or tools.
"""
import time
import os
import fcntl
from pathlib import Path
from openai import OpenAI

from .common import event, digest


class DeadlineExceeded(Exception):
    pass


class IncompleteResponse(Exception):
    pass


def throttle(interval):
    """One request admission clock shared by campaign threads and processes."""
    path = os.environ.get('REASONINGBANK_RATE_FILE')
    if not path or not interval:
        return
    with Path(path).open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        due = float(stream.read() or '0')
        time.sleep(max(0, due - time.time()))
        stream.seek(0); stream.truncate()
        stream.write(str(time.time() + interval)); stream.flush()


def is_quota_error(exc):
    return any(s in str(exc).lower() for s in [
        'insufficient_quota','budget exceeded','budget has been exceeded',
        'budget exhausted','payment required','exceeded your current quota'])


class Model:
    def __init__(self, config, out, phase, deadline, selected_memory='', client=None, hard_deadline=None):
        self.config=config
        self.out=out
        self.phase=phase
        self.deadline=deadline
        self.hard_deadline=hard_deadline
        self.phase_seconds=max(0, deadline-time.monotonic())
        self.timeout_recoveries=int(config.get('timeout_recoveries', 0))
        self.client=client or OpenAI(max_retries=0)
        self.cost=0.0  # No provider price assumed; budget is wall time/steps.
        self.n_calls=0
        self.selected_memory=selected_memory

    def query(self, messages, **kwargs):
        return {'content':self.complete(messages, self.config['temperature'])}

    def complete(self, messages, temperature):
        import json
        attempts = self.config.get('request_attempts', 3)
        for attempt in range(attempts):
            throttle(self.config.get('request_interval', 0))
            remaining=self.deadline-time.monotonic()-3
            if remaining<=0:
                raise DeadlineExceeded('Phase deadline reached')
            self.n_calls+=1
            from swarmevo.runtime.images import image_usage, content_text
            event(self.out,'model_request',phase=self.phase,attempt=attempt+1,
                  messages_sha256=digest(json.dumps(messages,ensure_ascii=False)),
                  message_chars=sum(len(content_text(m['content'])) for m in messages),
                  **image_usage(messages),
                  memory_supplied=bool(self.selected_memory),
                  memory_in_system=bool(self.selected_memory and self.selected_memory in messages[0]['content']),
                  timeout=min(remaining,self.config['api_timeout']),
                  model=self.config['model'],temperature=temperature,
                  max_tokens=self.config.get('learning_max_tokens', self.config['max_tokens']) if self.phase != 'solve' else self.config['max_tokens'],
                  thinking=self.config.get('thinking'))
            try:
                extra = {'extra_body': {'thinking': self.config['thinking']}} if 'thinking' in self.config else {}
                response=self.client.chat.completions.create(model=self.config['model'],messages=messages,
                    temperature=temperature,max_tokens=self.config.get('learning_max_tokens', self.config['max_tokens']) if self.phase != 'solve' else self.config['max_tokens'],
                    timeout=min(remaining,self.config['api_timeout']), **extra)
                text=response.choices[0].message.content or ''
                reasoning=getattr(response.choices[0].message, 'reasoning_content', None) or ''
                event(self.out,'model_response',phase=self.phase,usage=response.usage.model_dump() if response.usage else {},
                      finish_reason=response.choices[0].finish_reason,content=text,
                      response_id=response.id,response_model=response.model,
                      reasoning_chars=len(reasoning))
                if response.choices[0].finish_reason != 'stop' or not text.strip():
                    raise IncompleteResponse('Response did not complete normally')
            except Exception as exc:
                quota=is_quota_error(exc)
                event(self.out,'model_error',phase=self.phase,error=type(exc).__name__,quota_exhausted=quota)
                if quota or type(exc).__name__ not in {'RateLimitError','APITimeoutError','InternalServerError','APIConnectionError','IncompleteResponse'} or attempt==attempts-1:
                    raise
                if self.deadline-time.monotonic() <= 5 and self.timeout_recoveries > 0:
                    renewed=time.monotonic()+self.phase_seconds
                    if self.hard_deadline is not None:
                        renewed=min(renewed,self.hard_deadline)
                    if renewed > self.deadline:
                        self.timeout_recoveries-=1
                        self.deadline=renewed
                        event(self.out,'phase_timeout_recovery',phase=self.phase,
                              recoveries_remaining=self.timeout_recoveries,
                              renewed_seconds=max(0,renewed-time.monotonic()))
                delay=min(10*(2**attempt),max(0,self.deadline-time.monotonic()-5))
                if delay<=0:
                    raise DeadlineExceeded('No time for retry')
                event(self.out,'model_retry',phase=self.phase,delay_seconds=delay)
                time.sleep(delay)
                continue
            return text

    def get_template_vars(self):
        return {'model_name':self.config['model'],'n_model_calls':self.n_calls,'model_cost':self.cost}
