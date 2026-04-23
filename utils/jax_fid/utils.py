"""Compatibility shim for historical utility helpers."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from hashlib import md5
from urllib.request import urlopen

from utils.logging import log_for_all, log_for_0


def download(url, target_md5):
    cache_path = "/tmp/inception_params.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    i = 0
    log_for_0("Downloading inception checkpoint...")
    while i < 10:
        i += 1
        if i > 1:
            log_for_all(f"warning: retrying download {i}/10 ...")
        try:
            with urlopen(url) as resp:
                data = resp.read()
        except Exception as e:
            log_for_all(f"Failed to download {url}, error: {e}, retrying...")
            continue

        md5_hash = md5(data).hexdigest()
        if md5_hash != target_md5:
            log_for_all(f"Checksum mismatch, expected {target_md5}, got {md5_hash}, retrying...")
            continue
        try:
            params_dict = pickle.loads(data)
            log_for_0("Downloaded and verified inception checkpoint successfully.")
            out = Path(cache_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("wb") as f:
                pickle.dump(params_dict, f)
            return params_dict
        except Exception as e:
            log_for_all(f"Failed to load pickle data, error: {e}, retrying...")
            continue
    raise RuntimeError(f"Failed to download or validate the file after {i} attempts.")


def get(dictionary, key):
    if dictionary is None or key not in dictionary:
        return None
    return dictionary[key]
