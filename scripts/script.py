from pathlib import Path
import gradio as gr
import threading
import json
import re
import os

from modules.script_callbacks import on_app_started, on_infotext_pasted, on_before_image_saved, on_ui_settings
from modules import sd_vae, shared
import civitai.lib as civitai

types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint']
lock = threading.Lock()

base_model = {
    'SD 1': 'SD1',
    'SD 1.5': 'SD1',
    'SD 2': 'SD2',
    'SD 3': 'SD3',
    'SDXL': 'SDXL',
    'Pony': 'SDXL',
    'Illustrious': 'SDXL',
}

def get_sd_version(t: str):
    for k, v in base_model.items():
        if k in t: return v
    return ''

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

            data = {
                'activation text': ', '.join(r['trainedWords']),
                'sd version': get_sd_version(r['baseModel']),
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

def on_saved(params):
    additional_network_type_map = {'lora': 'LORA', 'hypernet': 'Hypernetwork'}
    additional_network_pattern = r'<(lora|hypernet):([a-zA-Z0-9_\.\-\s]+):([0-9.]+)(?:[:].*)?>'
    model_hash_pattern = r'Model hash: ([0-9a-fA-F]{10})'

    if 'parameters' not in params.pnginfo: return

    hashify_resources = shared.opts.data.get('civitai_hashify_resources', True)
    if not hashify_resources: return

    lines = params.pnginfo['parameters'].split('\n')
    generation_params = lines.pop()
    prompt_parts = '\n'.join(lines).split('Negative prompt:')
    prompt, negative_prompt = [s.strip() for s in prompt_parts[:2] + ['']*(2-len(prompt_parts))]

    resources = civitai.load_resource_list([])
    resource_hashes = {}

    if hashify_resources and sd_vae.loaded_vae_file is not None:
        vae_name = os.path.splitext(sd_vae.get_filename(sd_vae.loaded_vae_file))[0]
        vae_matches = [r for r in resources if r['type'] == 'VAE' and r['name'] == vae_name]
        if len(vae_matches) > 0:
            short_hash = vae_matches[0]['hash'][:10]
            resource_hashes['vae'] = short_hash

    embeddings = [r for r in resources if r['type'] == 'TextualInversion']
    for embedding in embeddings:
        embedding_name = embedding['name']
        embedding_pattern = re.compile(r'(?<![^\s:(|\[\]])' + re.escape(embedding_name) + r'(?![^\s:)|\[\]\,])', re.MULTILINE | re.IGNORECASE)

        match_prompt = embedding_pattern.search(prompt)
        match_negative = embedding_pattern.search(negative_prompt)
        if not match_prompt and not match_negative: continue

        short_hash = embedding['hash'][:10]
        resource_hashes[f'embed:{embedding_name}'] = short_hash

    network_matches = re.findall(additional_network_pattern, prompt)
    for match in network_matches:
        network_type, network_name, network_weight = match
        resource_type = additional_network_type_map[network_type]
        matching_resource = [r for r in resources if r['type'] == resource_type and (r['name'].lower() == network_name.lower() or r['name'].lower().split('-')[0] == network_name.lower())]
        if len(matching_resource) > 0:
            short_hash = matching_resource[0]['hash'][:10]
            resource_hashes[f'{network_type}:{network_name}'] = short_hash

    model_match = re.search(model_hash_pattern, generation_params)
    if hashify_resources and model_match:
        model_hash = model_match.group(1)
        matching_resource = [r for r in resources if r['type'] == 'Checkpoint' and r['hash'].startswith(model_hash)]
        if len(matching_resource) > 0:
            short_hash = matching_resource[0]['hash'][:10]
            resource_hashes['model'] = short_hash

    if len(resource_hashes) > 0:
        params.pnginfo['parameters'] += f', Hashes: {json.dumps(resource_hashes)}'

def on_pasted(infotext, params):
    model_hash = params['Model hash']
    model = civitai.get_model_by_hash(model_hash)
    if (model is None): return

def on_settings():
    section = ('civitai_extension', 'Civitai')
    shared.opts.add_option('civitai_api_key', shared.OptionInfo('', 'Your Civitai API Key', section=section))
    shared.opts.add_option('civitai_hashify_resources', shared.OptionInfo(True, 'Include resource hashes in image metadata (for resource auto-detection on Civitai)', section=section))
    shared.opts.add_option('civitai_folder_model', shared.OptionInfo('', 'Models directory (if not default)', section=section))
    shared.opts.add_option('civitai_folder_lora', shared.OptionInfo('', 'LoRA directory (if not default)', section=section))
    shared.opts.add_option('civitai_folder_lyco', shared.OptionInfo('', 'LyCORIS directory (if not default)', section=section))

on_app_started(app)
on_infotext_pasted(on_pasted)
on_before_image_saved(on_saved)
on_ui_settings(on_settings)