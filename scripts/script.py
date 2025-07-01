from pathlib import Path
import gradio as gr
import threading
import json

from modules.script_callbacks import on_app_started, on_ui_settings
from modules import sd_vae, shared

import civitai.lib as civitai

types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
lock = threading.Lock()

def get_resources(k: str):
    resources = civitai.load_resource_list()
    filtered = [r for r in resources if r['type'] in types]
    missing = [r for r in filtered if r[k] is False]
    hashes = [r['hash'] for r in missing]
    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results: return None, None, None
    return missing, hashes, results

def load_info():
    resources = civitai.load_resource_list()
    hashes = [r['hash'] for r in resources]
    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results: return

    N = 0
    map = {r['hash']: r for r in resources}

    baseList = {
        'SD 1': 'SD1',
        'SD 1.5': 'SD1',
        'SD 2': 'SD2',
        'SD 3': 'SD3',
        'SDXL': 'SDXL',
        'Pony': 'SDXL',
        'Illustrious': 'SDXL',
    }

    for r in results:
        if r is None: continue

        for f in r['files']:
            if 'hashes' not in f or 'SHA256' not in f['hashes']: continue

            sha256 = f['hashes']['SHA256'].lower()
            if sha256 not in map: continue

            res = map[sha256]
            infotags = Path(res['path']).with_suffix('.json')

            trainedWords = ', '.join(r.get('trainedWords', []))
            baseModel = next((v for k, v in baseList.items() if k in r.get('baseModel', '')), '')

            data = {
                'activation text': trainedWords,
                'sd version': baseModel,
                'modelId': r['modelId'],
                'modelVersionId': r['id'],
                'sha256': sha256.upper()
            }

            if infotags.exists():
                try:
                    j = json.loads(infotags.read_text(encoding='utf-8'))
                except Exception:
                    j = {}

                updated = False

                if 'activation text' not in j:
                    j['activation text'] = trainedWords
                    updated = True

                if 'sd version' not in j or j['sd version'] != baseModel:
                    j['sd version'] = baseModel
                    updated = True

                if updated:
                    infotags.write_text(json.dumps(j, indent=4), encoding='utf-8')
                    N += 1
            else:
                infotags.write_text(json.dumps(data, indent=4), encoding='utf-8')
                N += 1

    if N > 0: civitai.log(f'Updated {N} info files')

def load_preview():
    hashes = [r['hash'] for r in civitai.load_resource_list() if r['type'] in types and not r['hasPreview']]
    results = civitai.get_all_by_hash_with_cache(hashes)
    if not results: return

    N = 0

    for r in results:
        if r is None: continue

        for f in r['files']:
            if not 'hashes' in f or not 'SHA256' in f['hashes']: continue
            sha256 = f['hashes']['SHA256']
            if sha256.lower() not in hashes: continue
            img = r['images']
            if len(img) == 0: continue
            preview = next((p for p in img if not p['url'].lower().endswith(('.mp4', '.gif'))), None)
            if preview is None: continue
            url = preview['url']
            civitai.update_resource_preview(sha256, url)
            N += 1

    if N > 0: civitai.log(f'Updated {N} preview images')

def run_load_info():
    with lock: load_info()

def run_load_preview():
    with lock: load_preview()

def app(_: gr.Blocks, app):
    civitai.log('Check resources for missing info files and previews')

    thread1 = threading.Thread(target=run_load_info)
    thread2 = threading.Thread(target=run_load_preview)

    thread1.start()
    thread2.start()

def on_settings():
    section = ('civitai_extension', 'Civitai')
    shared.opts.add_option('civitai_api_key', shared.OptionInfo('', 'Your Civitai API Key', section=section))
    shared.opts.add_option('civitai_hashify_resources', shared.OptionInfo(True, 'Include resource hashes in image metadata (for resource auto-detection on Civitai)', section=section))
    shared.opts.add_option('civitai_folder_model', shared.OptionInfo('', 'Models directory (if not default)', section=section))
    shared.opts.add_option('civitai_folder_lora', shared.OptionInfo('', 'LoRA directory (if not default)', section=section))
    shared.opts.add_option('civitai_folder_lyco', shared.OptionInfo('', 'LyCORIS directory (if not default)', section=section))

on_app_started(app)
on_ui_settings(on_settings)