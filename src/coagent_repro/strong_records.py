"""Same fixed-record experiment with disclosed large-model transport presets."""
import argparse
import hashlib
from pathlib import Path
from . import replicate_records as harness
from benign_record_feedback import lab
from benign_llm_orca import study

MODELS = ('qwen3:32b', 'qwq:32b')
OriginalBackend = lab.RecordBackend
OriginalFuse = study.Fuse
original_hashes = harness.source_hashes


class LargeBackend(OriginalBackend):
    def __init__(self, port, fuse, model_name):
        super().__init__(port, fuse, model_name)
        self.identity.update(max_output_tokens=512 if model_name=='qwq:32b' else 64,
            selector_max_output_tokens=512 if model_name=='qwq:32b' else 16,
            reasoning_preset='model-default' if model_name=='qwq:32b' else 'non-thinking',
            preset_scope='same for all four conditions; not tuned on evaluation results')
    def request(self, path, payload=None):
        if path=='/api/generate' and self.model_name=='qwq:32b':
            payload = dict(payload)
            payload.pop('think',None)
            payload['options']={**payload['options'],'num_predict':512}
        return super().request(path,payload)


class LargeFuse(OriginalFuse):
    def __init__(self, max_calls, max_tokens, max_seconds):
        super().__init__(max_calls=max_calls,max_tokens=400000,
                         max_seconds=900 if max_calls==350 else 1800)


def hashes():
    return {**original_hashes(),Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--mode',choices=('smoke','replication'),required=True)
    p.add_argument('--seed',type=int,choices=harness.SEEDS,required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    harness.MODELS=MODELS
    harness.source_hashes=hashes
    lab.partitions=harness.partitions
    lab.RecordBackend=LargeBackend
    study.Fuse=LargeFuse
    harness.run(args)


if __name__ == '__main__':
    main()
