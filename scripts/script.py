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
    missing, hashes, results = get_resources('hasInfo')
    if results is None: return

    N = 0

    for r in results:
        if (r is None): continue

        for f in r['files']:
            if not 'hashes' in f or not 'SHA256' in f['hashes']: continue
            h = f['hashes']['SHA256']
            if h.lower() not in hashes: continue

            base_model = {
                'SD 1': 'SD1',
                'SD 1.5': 'SD1',
                'SD 2': 'SD2',
                'SD 3': 'SD3',
                'SDXL': 'SDXL',
                'Pony': 'SDXL',
                'Illustrious': 'SDXL',
            }

            data = {
                'activation text': ', '.join(r['trainedWords']),
                'sd version': next((v for k, v in base_model.items() if k in r['baseModel']), ''),
                'modelId': r['modelId'],
                'modelVersionId': r['id'],
                'sha256': h.upper()
            }

            v = [r for r in missing if h.lower() == r['hash']]
            if len(v) == 0: continue
            for r in v: Path(r['path']).with_suffix('.json').write_text(json.dumps(data, indent=4), encoding='utf-8')
            N += 1

    if N > 0: civitai.log(f'Updated {N} info files')

def load_preview():
    missing, hashes, results = get_resources('hasPreview')
    if results is None: return

    N = 0

    for r in results:
        if (r is None): continue

        for f in r['files']:
            if not 'hashes' in f or not 'SHA256' in f['hashes']: continue
            h = f['hashes']['SHA256']
            if h.lower() not in hashes: continue
            img = r['images']
            if len(img) == 0: continue
            p = next((p for p in img if not p['url'].lower().endswith(('.mp4', '.gif'))), None)
            if p is None: continue
            url = p['url']
            civitai.update_resource_preview(h, url)
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