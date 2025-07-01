from modules import shared, sd_models, sd_vae, hashes, ui_extra_networks, cache
from modules.paths import models_path
from datetime import datetime
from pathlib import Path
from typing import List
from tqdm import tqdm
from PIL import Image
import tempfile
import requests
import shutil
import glob
import json
import time
import os
import io

base_url = 'https://civitai.com/api/v1'
user_agent = 'CivitaiLink:Automatic1111'
download_chunk_size = 8192
civil_ai_api_cache = cache.cache('civil_ai_api_sha256')
KAGGLE = 'KAGGLE_DATA_PROXY_TOKEN' in os.environ

RST = '\033[0m'
BLUE = '\033[38;5;39m'

resources = []

def log(message):
    print(f'{BLUE}●{RST} Civitai: {message}')

def req(endpoint, method='GET', data=None, params=None, headers=None):
    if headers is None: headers = {}
    headers['User-Agent'] = user_agent
    api_key = shared.opts.data.get("civitai_api_key", None)
    if api_key is not None: headers['Authorization'] = f'Bearer {api_key}'
    if data is not None: headers['Content-Type'] = 'application/json'; data = json.dumps(data)
    if not endpoint.startswith('/'): endpoint = '/' + endpoint
    if params is None: params = {}
    response = requests.request(method, base_url+endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200: raise Exception(f'Error: {response.status_code} {response.text}')
    return response.json()

def get_all_by_hash(hashes: List[str]):
    response = req(f"/model-versions/by-hash", method='POST', data=hashes)
    return response

def get_lora_dir():
    return shared.cmd_opts.lora_dir

def get_locon_dir():
    try: return shared.cmd_opts.lyco_dir or get_lora_dir()
    except AttributeError: return get_lora_dir()

def get_model_dir():
    return shared.cmd_opts.ckpt_dir or sd_models.model_path

def get_automatic_type(file_type: str):
    if file_type == 'Hypernetwork': return 'hypernet'
    return file_type.lower()

def get_automatic_name(file_type: str, filename: str, folder: str):
    path = Path(filename).resolve()
    folder_path = Path(folder).resolve()

    try: fullname = path.relative_to(folder_path)
    except ValueError: fullname = path.name

    if file_type == 'Checkpoint': return str(fullname)
    return str(fullname.with_suffix(''))

def has_preview(filename: str):
    ui_extra_networks.allowed_preview_extensions()
    preview_exts = ui_extra_networks.allowed_preview_extensions()
    preview_exts = [*preview_exts, *["preview." + x for x in preview_exts]]
    for ext in preview_exts:
        if Path(filename).with_suffix(f'.{ext}').exists(): return True
    return False

def has_info(filename: str):
    return Path(filename).with_suffix('.json').exists()

def get_resources_in_folder(file_type, folder, exts=None, exts_exclude=None):
    exts = exts or []
    exts_exclude = exts_exclude or []
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)

    automatic_type = get_automatic_type(file_type)
    candidates = [f for ext in exts for f in folder.rglob(f'*.{ext}') if not any(str(f).endswith(e) for e in exts_exclude)]

    _resources = []
    cmd_opts_no_hashing = shared.cmd_opts.no_hashing
    shared.cmd_opts.no_hashing = False
    try:
        for f in sorted(candidates):
            if f.is_dir(): continue

            name = f.stem
            automatic_name = get_automatic_name(file_type, str(f), str(folder))
            file_hash = hashes.sha256(str(f), f"{automatic_type}/{automatic_name}")

            _resources.append({
                'type': file_type,
                'name': name,
                'hash': file_hash,
                'path': str(f),
                'hasPreview': has_preview(str(f)),
                'hasInfo': has_info(str(f))
            })

    finally: shared.cmd_opts.no_hashing = cmd_opts_no_hashing

    return _resources

def get_all_by_hash_with_cache(file_hashes: List[str]):
    missing_info_hashes = [file_hash for file_hash in file_hashes if file_hash not in civil_ai_api_cache]
    new_results = []

    try:
        for i in range(0, len(missing_info_hashes), 100):
            batch = missing_info_hashes[i:i + 100]
            new_results.extend(get_all_by_hash(batch))
    except Exception as e: raise e

    new_results = sorted(new_results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    found_info_hashes = set()

    for new_metadata in new_results:
        for file in new_metadata['files']:
            file_hash = file['hashes']['SHA256'].lower()
            civil_ai_api_cache[file_hash] = new_metadata
            found_info_hashes.add(file_hash)

    for file_hash in set(missing_info_hashes) - found_info_hashes: civil_ai_api_cache[file_hash] = None

    final_results = []

    for h in file_hashes:
        cached = civil_ai_api_cache.get(h)
        if cached: final_results.append(cached)

    return final_results

def load_resource_list(types=None):
    global resources
    if types is None:
        types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint', 'VAE', 'Controlnet', 'Upscaler']

    res = [r for r in resources if r['type'] not in types]

    folders = {
        'LORA': Path(get_lora_dir()),
        'LoCon': Path(get_locon_dir()),
        'Hypernetwork': Path(shared.cmd_opts.hypernetwork_dir),
        'TextualInversion': Path(shared.cmd_opts.embeddings_dir),
        'Checkpoint': Path(get_model_dir()),
        'Controlnet': Path(models_path) / 'ControlNet',
        'Upscaler': Path(models_path) / 'ESRGAN',
        'VAE1': Path(get_model_dir()),
        'VAE2': Path(sd_vae.vae_path),
    }

    if 'LORA' in types:
        res += get_resources_in_folder('LORA', folders['LORA'], ['pt', 'safetensors', 'ckpt'])
    if 'LoCon' in types and folders['LORA'] != folders['LoCon']:
        res += get_resources_in_folder('LoCon', folders['LoCon'], ['pt', 'safetensors', 'ckpt'])
    if 'Hypernetwork' in types:
        res += get_resources_in_folder('Hypernetwork', folders['Hypernetwork'], ['pt', 'safetensors', 'ckpt'])
    if 'TextualInversion' in types:
        res += get_resources_in_folder('TextualInversion', folders['TextualInversion'], ['pt', 'bin', 'safetensors'])
    if 'Checkpoint' in types:
        res += get_resources_in_folder('Checkpoint', folders['Checkpoint'], ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Controlnet' in types:
        res += get_resources_in_folder('Controlnet', folders['Controlnet'], ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Upscaler' in types:
        res += get_resources_in_folder('Upscaler', folders['Upscaler'], ['safetensors', 'ckpt', 'pt'])
    if 'VAE' in types:
        res += get_resources_in_folder('VAE', folders['VAE1'], ['vae.pt', 'vae.safetensors', 'vae.ckpt'])
        res += get_resources_in_folder('VAE', folders['VAE2'], ['pt', 'safetensors', 'ckpt'])

    resources = res
    return resources

def get_model_by_hash(file_hash: str):
    if found := [info for info in sd_models.checkpoints_list.values() if file_hash == info.sha256 or file_hash == info.shorthash or file_hash == info.hash]:
        return found[0]

def get_resource_by_hash(hash: str):
    resources = load_resource_list([])

    found = [resource for resource in resources if hash.lower() == resource['hash'] and ('downloading' not in resource or resource['downloading'] != True)]
    if found:
        return found[0]

    return None

def resizer(b, size=512):
    i = Image.open(io.BytesIO(b))
    w, h = i.size
    s = (size, int(h * size / w)) if w > h else (int(w * size / h), size)
    o = io.BytesIO()
    i.resize(s, Image.LANCZOS).save(o, format='PNG')
    o.seek(0)
    return o

def download_preview(url, dest_path, on_progress=None):
    dest = Path(dest_path).expanduser()
    if dest.exists(): return

    response = requests.get(url, stream=True, headers={"User-Agent": user_agent})
    total = int(response.headers.get('content-length', 0))
    start_time = time.time()

    try:
        image_data = bytearray()
        current = 0
        for data in response.iter_content(chunk_size=download_chunk_size):
            image_data.extend(data)
            current += len(data)
            if on_progress is not None:
                should_stop = on_progress(current, total, start_time)
                if should_stop:
                    raise Exception("Download cancelled")

        resized = resizer(image_data)

        if KAGGLE:
            import sd_image_encryption # type: ignore

            img = Image.open(resized)
            imginfo = img.info or {}
            if not all(k in imginfo for k in ['Encrypt', 'EncryptPwdSha']):
                sd_image_encryption.EncryptedImage.from_image(img).save(dest)
        else:
            dest.write_bytes(resized.read())

    except Exception as e:
        print(f"Preview failed: {dest} : {e}")
        if dest.exists():
            dest.unlink()

def update_resource_preview(hash: str, preview_url: str):
    for res in [r for r in load_resource_list([]) if r['hash'] == hash.lower()]:
        preview_path = Path(res['path']).with_suffix('.preview.png')
        download_preview(preview_url, str(preview_path))