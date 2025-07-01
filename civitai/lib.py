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

def log(message):
    print(f'{BLUE}●{RST} Civitai: {message}')

def req(endpoint, method='GET', data=None, params=None, headers=None):
    """Make a request to the Civitai API."""
    if headers is None:
        headers = {}
    headers['User-Agent'] = user_agent
    api_key = shared.opts.data.get("civitai_api_key", None)
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'
    if data is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    if params is None:
        params = {}
    response = requests.request(method, base_url+endpoint, data=data, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(f'Error: {response.status_code} {response.text}')
    return response.json()

def get_models(query, creator, tag, type, page=1, page_size=20, sort='Most Downloaded', period='AllTime'):
    """Get a list of models from the Civitai API."""
    response = req('/models', params={
        'query': query,
        'username': creator,
        'tag': tag,
        'type': type,
        'sort': sort,
        'period': period,
        'page': page,
        'pageSize': page_size,
    })
    return response

def get_all_by_hash(hashes: List[str]):
    response = req(f"/model-versions/by-hash", method='POST', data=hashes)
    return response

def get_model_version(id):
    """Get a model version from the Civitai API."""
    response = req('/model-versions/'+id)
    return response

def get_model_version_by_hash(hash: str):
    response = req(f"/model-versions/by-hash/{hash}")
    return response

def get_creators(query, page=1, page_size=20):
    """Get a list of creators from the Civitai API."""
    response = req('/creators', params={
        'query': query,
        'page': page,
        'pageSize': page_size
    })
    return response

def get_tags(query, page=1, page_size=20):
    """Get a list of tags from the Civitai API."""
    response = req('/tags', params={
        'query': query,
        'page': page,
        'pageSize': page_size
    })
    return response

def get_lora_dir():
    return shared.cmd_opts.lora_dir


def get_locon_dir():
    try:
        return shared.cmd_opts.lyco_dir or get_lora_dir()
    except AttributeError:
        return get_lora_dir()


def get_model_dir():
    return shared.cmd_opts.ckpt_dir or sd_models.model_path

def get_automatic_type(file_type: str):
    if file_type == 'Hypernetwork':
        return 'hypernet'
    return file_type.lower()


def get_automatic_name(file_type: str, filename: str, folder: str):
    abspath = os.path.abspath(filename)
    if abspath.startswith(folder):
        fullname = abspath.replace(folder, '')
    else:
        fullname = os.path.basename(filename)

    if fullname.startswith("\\") or fullname.startswith("/"):
        fullname = fullname[1:]

    if file_type == 'Checkpoint':
        return fullname
    return os.path.splitext(fullname)[0]

def has_preview(filename: str):
    ui_extra_networks.allowed_preview_extensions()
    preview_exts = ui_extra_networks.allowed_preview_extensions()
    preview_exts = [*preview_exts, *["preview." + x for x in preview_exts]]
    for ext in preview_exts:
        if os.path.exists(os.path.splitext(filename)[0] + '.' + ext):
            return True
    return False


def has_info(filename: str):
    return os.path.isfile(os.path.splitext(filename)[0] + '.json')

def get_resources_in_folder(file_type, folder, exts=None, exts_exclude=None):
    if exts_exclude is None:
        exts_exclude = []
    if exts is None:
        exts = []
    _resources = []
    os.makedirs(folder, exist_ok=True)

    candidates = []
    for ext in exts:
        candidates += glob.glob(os.path.join(folder, '**/*.' + ext), recursive=True)
    for ext in exts_exclude:
        candidates = [x for x in candidates if not x.endswith(ext)]

    folder = os.path.abspath(folder)
    automatic_type = get_automatic_type(file_type)

    cmd_opts_no_hashing = shared.cmd_opts.no_hashing
    shared.cmd_opts.no_hashing = False
    try:
        for filename in sorted(candidates):
            if os.path.isdir(filename):
                continue

            name = os.path.splitext(os.path.basename(filename))[0]
            automatic_name = get_automatic_name(file_type, filename, folder)
            file_hash = hashes.sha256(filename, f"{automatic_type}/{automatic_name}")

            _resources.append({'type': file_type, 'name': name, 'hash': file_hash, 'path': filename, 'hasPreview': has_preview(filename), 'hasInfo': has_info(filename)})
    finally:
        shared.cmd_opts.no_hashing = cmd_opts_no_hashing
    return _resources

def get_all_by_hash_with_cache(file_hashes: List[str]):
    """"Un-finished function"""

    # cached_info_hashes = [file_hash for file_hash in file_hashes if file_hash in metadata_cache_dict]
    missing_info_hashes = [file_hash for file_hash in file_hashes if file_hash not in civil_ai_api_cache]
    new_results = []
    try:
        for i in range(0, len(missing_info_hashes), 100):
            batch = missing_info_hashes[i:i + 100]
            new_results.extend(get_all_by_hash(batch))

    except Exception as e:
        raise e

    new_results = sorted(new_results, key=lambda x: datetime.fromisoformat(x['createdAt'].rstrip('Z')), reverse=True)

    found_info_hashes = set()
    for new_metadata in new_results:
        for file in new_metadata['files']:
            file_hash = file['hashes']['SHA256'].lower()
            found_info_hashes.add(file_hash)
    for file_hash in set(missing_info_hashes) - found_info_hashes:
        if file_hash not in civil_ai_api_cache:
            civil_ai_api_cache[file_hash] = None
    return new_results

resources = []

def load_resource_list(types=None):
    global resources

    if types is None: types = ['LORA', 'LoCon', 'Hypernetwork', 'TextualInversion', 'Checkpoint', 'VAE', 'Controlnet', 'Upscaler']
    lora_dir = get_lora_dir()

    if 'LORA' in types:
        resources = [r for r in resources if r['type'] != 'LORA']
        resources += get_resources_in_folder('LORA', lora_dir, ['pt', 'safetensors', 'ckpt'])
    if 'LoCon' in types:
        lycoris_dir = get_locon_dir()
        if lora_dir != lycoris_dir:
            resources = [r for r in resources if r['type'] != 'LoCon']
            resources += get_resources_in_folder('LoCon', get_locon_dir(), ['pt', 'safetensors', 'ckpt'])
    if 'Hypernetwork' in types:
        resources = [r for r in resources if r['type'] != 'Hypernetwork']
        resources += get_resources_in_folder('Hypernetwork', shared.cmd_opts.hypernetwork_dir, ['pt', 'safetensors', 'ckpt'])
    if 'TextualInversion' in types:
        resources = [r for r in resources if r['type'] != 'TextualInversion']
        resources += get_resources_in_folder('TextualInversion', shared.cmd_opts.embeddings_dir, ['pt', 'bin', 'safetensors'])
    if 'Checkpoint' in types:
        resources = [r for r in resources if r['type'] != 'Checkpoint']
        resources += get_resources_in_folder('Checkpoint', get_model_dir(), ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Controlnet' in types:
        resources = [r for r in resources if r['type'] != 'Controlnet']
        resources += get_resources_in_folder('Controlnet', os.path.join(models_path, "ControlNet"), ['safetensors', 'ckpt'], ['vae.safetensors', 'vae.ckpt'])
    if 'Upscaler' in types:
        resources = [r for r in resources if r['type'] != 'Upscaler']
        resources += get_resources_in_folder('Upscaler', os.path.join(models_path, "ESRGAN"), ['safetensors', 'ckpt', 'pt'])
    if 'VAE' in types:
        resources = [r for r in resources if r['type'] != 'VAE']
        resources += get_resources_in_folder('VAE', get_model_dir(), ['vae.pt', 'vae.safetensors', 'vae.ckpt'])
        resources += get_resources_in_folder('VAE', sd_vae.vae_path, ['pt', 'safetensors', 'ckpt'])

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
    resources = load_resource_list([])
    matches = [resource for resource in resources if hash.lower() == resource['hash']]
    if len(matches) == 0: return

    for resource in matches:
        preview_path = os.path.splitext(resource['path'])[0] + '.preview.png'
        download_preview(preview_url, preview_path)