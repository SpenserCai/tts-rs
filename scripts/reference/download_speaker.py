"""Download only Qwen speaker safetensors ranges from the pinned official model.

HTTP Range responses are checked strictly. A server silently returning the whole
multi-gigabyte file is rejected. The resulting subset is for local validation;
weights are never checked into this repository.
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

from common import PINS, sha256, write_json


def get_range(url: str, first: int, last: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        expected = f"bytes {first}-{last}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise RuntimeError(f"Server did not honor the exact byte range {first}-{last}")
        data = response.read(last - first + 2)
    if len(data) != last - first + 1:
        raise RuntimeError("Truncated or oversized HTTP range response")
    return data


def download(output: Path) -> Path:
    pin = PINS["qwen"]
    output.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{pin['model_id']}/resolve/{pin['model_revision']}/model.safetensors"
    header_size = struct.unpack("<Q", get_range(url, 0, 7))[0]
    if header_size > 16 * 1024 * 1024:
        raise RuntimeError("Implausibly large safetensors header")
    header = json.loads(get_range(url, 8, 7 + header_size))
    selected = sorted(
        ((name, info) for name, info in header.items() if name.startswith("speaker_encoder.")),
        key=lambda entry: entry[1]["data_offsets"][0],
    )
    if not selected:
        raise RuntimeError("Pinned official checkpoint has no speaker_encoder tensors")
    # Adjacent speaker tensors occupy a compact span in the official checkpoint.
    first = selected[0][1]["data_offsets"][0]
    last = max(info["data_offsets"][1] for _, info in selected)
    if last - first > 256 * 1024 * 1024:
        raise RuntimeError("Speaker tensor span exceeds 256 MiB; inspect the checkpoint layout")
    print(f"Downloading {len(selected)} speaker tensors ({last - first} bytes)", flush=True)
    raw = get_range(url, 8 + header_size + first, 8 + header_size + last - 1)
    subset_header = {}
    pieces = []
    offset = 0
    for name, info in selected:
        start, end = info["data_offsets"]
        piece = raw[start - first:end - first]
        subset_header[name.removeprefix("speaker_encoder.")] = {
            "dtype": info["dtype"], "shape": info["shape"],
            "data_offsets": [offset, offset + len(piece)],
        }
        pieces.append(piece)
        offset += len(piece)
    subset_header["__metadata__"] = {"source": pin["model_id"], "revision": pin["model_revision"]}
    encoded = json.dumps(subset_header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    path = output / "speaker-original.safetensors"
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for piece in pieces:
            stream.write(piece)
    config_url = f"https://huggingface.co/{pin['model_id']}/resolve/{pin['model_revision']}/config.json"
    with urllib.request.urlopen(config_url, timeout=120) as response:
        config = json.load(response)
    write_json(output / "model-config.json", config)
    write_json(output / "download.json", {
        "schema_version": 1, "source_url": url, "model_revision": pin["model_revision"],
        "tensors": len(selected), "bytes_downloaded": len(raw),
        "subset_sha256": sha256(path), "weights_license": "Apache-2.0 (verify model card before redistribution)",
    })
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(download(arguments.output))
