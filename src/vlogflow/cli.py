from __future__ import annotations
import argparse
import json
import shutil
import sys
import webbrowser
from pathlib import Path

from .config import data_root,load_settings,save_settings
from .models import download_model


def main():
    p=argparse.ArgumentParser(prog='vlogflow',description='Review-first local vlog creation')
    sub=p.add_subparsers(dest='command')
    serve=sub.add_parser('serve'); serve.add_argument('--port',type=int,default=7860); serve.add_argument('--no-browser',action='store_true')
    models=sub.add_parser('models'); ms=models.add_subparsers(dest='action',required=True)
    download=ms.add_parser('download'); download.add_argument('kind',choices=['qwen-mlx','asr','qwen-torch'])
    sub.add_parser('doctor')
    a=p.parse_args(); root=data_root()
    if a.command=='models':
        path=download_model(root,a.kind); settings=load_settings(root)
        if a.kind=='asr': settings.asr_model=str(path)
        else: settings.tts_model=str(path); settings.tts_backend='mlx' if a.kind=='qwen-mlx' else 'torch'
        save_settings(root,settings); print(f'Verified model saved to {path}'); return
    if a.command=='doctor':
        import importlib.util
        import psutil
        settings=load_settings(root)
        result=dict(python=sys.version.split()[0],data_directory=str(root),ffmpeg=shutil.which(settings.ffmpeg),ffprobe=shutil.which(settings.ffprobe),
                    available_memory_gib=round(psutil.virtual_memory().available/1024**3,2),asr_installed=bool(importlib.util.find_spec('faster_whisper')),
                    mlx_installed=bool(importlib.util.find_spec('mlx_audio')),tts_backend=settings.tts_backend)
        print(json.dumps(result,indent=2)); return
    if a.command not in ('serve',None): p.error('Unknown command')
    import uvicorn
    from filelock import FileLock,Timeout
    from .api import create_app
    port=getattr(a,'port',7860)
    try:
        with FileLock(str(root/'server.lock'),timeout=0):
            url=f'http://127.0.0.1:{port}'
            print(f'VlogFlow: {url}\nData: {root}',flush=True)
            if not getattr(a,'no_browser',False):
                import threading
                threading.Timer(1.2,lambda:webbrowser.open(url)).start()
            uvicorn.run(create_app(root),host='127.0.0.1',port=port,log_level='warning')
    except Timeout:
        p.error('Another VlogFlow instance is using this data directory. Open its browser tab or stop it first.')


if __name__=='__main__': main()
