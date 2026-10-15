"""Isolated V3.2 transport/budget overlay; feedback, rewards and tasks unchanged."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from . import replicate_records as harness
from benign_record_feedback import lab
from benign_llm_orca import study

MODELS = ('deepseek-v3.2:q4km-local',)
OriginalBackend, OriginalFuse = lab.RecordBackend, study.Fuse
original_hashes = harness.source_hashes
ROOT = Path(__file__).resolve().parent


class LargeFuse(OriginalFuse):
    def __init__(self, max_calls, max_tokens, max_seconds):
        super().__init__(max_calls, min(max_tokens, 400000), min(max_seconds, 1800))
        # Explicit hardware-runtime envelope only; token/call ceilings unchanged.
        self.max_seconds = 1800 if max_calls == 9 else 3600 if max_calls == 350 else 10800


class LargeBackend(OriginalBackend):
    def __init__(self, port, fuse, model_name):
        super().__init__(port, fuse, model_name)
        weights = json.loads((ROOT/'weights_manifest.json').read_text())
        gpu_count = int(os.environ.get('V32_GPU_COUNT', '0'))
        layers = int(os.environ.get('V32_GPU_LAYERS', '0'))
        if (gpu_count, layers) not in ((2, 36), (4, 40), (4, 20)):
            raise ValueError('An audited two-H200 or four-A100 allocation is required')
        self.identity.update(weight_repository=weights['repository'], weight_revision=weights['revision'],
            quantization=weights['quantization'], weight_total_bytes=weights['total_bytes'],
            weight_manifest_sha256=hashlib.sha256((ROOT/'weights_manifest.json').read_bytes()).hexdigest(),
            reasoning_preset='non-thinking', gpu_offload_layers=layers, cpu_threads=16,
            cpu_offload_permitted=True, allocated_gpus=gpu_count, cold_start_timeout_seconds=900,
            regular_timeout_seconds=120, preset_scope='identical across four conditions; not accuracy-tuned')

    def request(self, path, payload=None):
        if path != '/api/generate':
            return super().request(path, payload)
        payload = dict(payload)
        payload['options'] = {**payload['options'], 'num_thread':16,
                              'num_gpu':self.identity['gpu_offload_layers']}
        payload['keep_alive'] = '13h'
        # Keep the existing prompts, schema, thinking flag and 64/16-token caps.
        wait = 900 if self.fuse.calls == 1 else 120
        req = urllib.request.Request(self.url+path, data=json.dumps(payload).encode(),
                                     headers={'Content-Type':'application/json'})
        started = time.monotonic()
        chunks, byte_count = [], 0
        try:
            with self.opener.open(req, timeout=wait) as response:
                for index, line in enumerate(response):
                    byte_count += len(line)
                    if byte_count > 1000000 or index >= 10000 or time.monotonic()-started > wait:
                        raise ValueError('Bounded V3.2 streaming budget exceeded')
                    frame = json.loads(line)
                    if frame.get('error'):
                        raise ValueError('Local V3.2 inference error: '+str(frame['error'])[:300])
                    text = frame.get('response', '')
                    if not isinstance(text, str):
                        raise ValueError('Invalid response text')
                    chunks.append(text)
                    if frame.get('done') is True:
                        return {**frame, 'response':''.join(chunks)}
            raise ValueError('Stream ended without final actual token counts')
        except Exception as error:
            self.failed_inference = {'response_chunks':chunks, 'final_frame':False,
                                     'error':str(error), 'retry':False}
            raise


def hashes():
    return {**original_hashes(), Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'weights_manifest.json':hashlib.sha256((ROOT/'weights_manifest.json').read_bytes()).hexdigest()}


def configure():
    harness.MODELS = MODELS
    harness.source_hashes = hashes
    lab.partitions = harness.partitions
    lab.RecordBackend = LargeBackend
    study.SUPPORTED_MODELS = MODELS
    study.Fuse = LargeFuse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--mode', choices=('smoke','replication'), required=True)
    p.add_argument('--seed', type=int, choices=harness.SEEDS, required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    configure()
    harness.run(args)


if __name__ == '__main__':
    main()
