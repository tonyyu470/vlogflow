from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from .config import load_settings
from .pipeline import run_job
from .processes import Cancelled,ResourceLimit,Runner
from .store import Conflict

ACTIVE=('queued','running')
KINDS={'import','analyze','reference','subtitles','render','audition'}


class Jobs:
    def __init__(self,store):
        self.store=store; store.recover()
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='vlogflow')
        self.lock=threading.Lock(); self.stopping=threading.Event()

    def submit(self,project_id,kind,payload=None):
        if kind not in KINDS: raise ValueError('Unknown job type')
        with self.lock:
            if any(j['status'] in ACTIVE for j in self.store.jobs(project_id)):
                raise Conflict('A task is already active in this project. Cancel it or wait for it to finish.')
            j=self.store.create_job(project_id,kind)
            j=self.store.update_job(j['id'],payload=payload or {})
            settings=load_settings(self.store.root)
            self.pool.submit(self._run,j,settings)
            return j

    def _run(self,j,settings):
        def cancelled(): return self.stopping.is_set() or self.store.job(j['id'])['cancel_requested']
        def phase(value): self.store.update_job(j['id'],phase=value)
        runner=Runner(self.store.root/'projects'/j['project_id']/'logs'/j['id'],cancelled,phase)
        try:
            runner.check(); self.store.update_job(j['id'],status='running',phase=j['kind'])
            changes,status=run_job(j,settings,self.store.root,runner)
            runner.check(); self.store.publish(j['id'],changes,status)
        except Cancelled as e:
            self.store.update_job(j['id'],status='cancelled',phase='cancelled',error=str(e))
        except ResourceLimit as e:
            self.store.update_job(j['id'],status='resource_wait',phase='resource_wait',error=str(e))
        except Exception as e:
            error=str(e)
            if settings.api_key: error=error.replace(settings.api_key,'[redacted]')
            self.store.update_job(j['id'],status='failed',error=error[:4000])

    def cancel(self,key):
        j=self.store.job(key)
        if j['status'] in ACTIVE:
            return self.store.update_job(key,cancel_requested=True)
        return j

    def retry(self,key):
        j=self.store.job(key)
        return self.submit(j['project_id'],j['kind'],j.get('payload',{}))

    def close(self):
        self.stopping.set(); self.pool.shutdown(wait=True,cancel_futures=False)


def public_job(j):
    return {k:v for k,v in j.items() if k not in ('snapshot','payload')}
