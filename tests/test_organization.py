"""Dataset, model-index and path checks; synthetic fixtures only."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from Experiment import layout
from benign_llm_orca import study
from coagent_repro import replicate_records as harness
from release_tools.export_synthetic_data import payloads
from release_tools.check_release import check

ROOT = Path(__file__).resolve().parents[1]


def test_export_matches_built_in_partition_byte_for_byte():
    for name, content in payloads().items():
        assert (ROOT / 'datasets' / 'synthetic' / name).read_bytes() == content


def test_public_partition_labels_and_hashes():
    directory = ROOT / 'datasets' / 'synthetic'
    manifest = json.loads((directory / 'manifest.json').read_text())
    train = [json.loads(line) for line in (directory / 'train.jsonl').read_text().splitlines()]
    test = [json.loads(line) for line in (directory / 'test.jsonl').read_text().splitlines()]
    assert len(train) == 18 and len(test) == 60
    assert not {r['source'] for r in train} & {r['source'] for r in test}
    assert manifest['labels_are_verifier_only'] is True
    assert manifest['partition_digest'] == study.digest([(p, t.__dict__, f) for p, t, f in harness.partitions(False)])
    for name, info in manifest['files'].items():
        raw = (directory / name).read_bytes()
        assert len(raw) == info['bytes'] and hashlib.sha256(raw).hexdigest() == info['sha256']


def test_model_registry_matches_adapter_allowlists():
    registry = json.loads((ROOT / 'models' / 'registry.json').read_text())
    assert registry['contains_weights'] is False
    observed = set()
    for item in registry['models']:
        module = importlib.import_module(item['module'])
        assert item['tag'] in module.MODELS
        observed.add(item['tag'])
    assert len(observed) == 7


def test_protocol_index_matches_actual_partition():
    spec = json.loads((ROOT / 'experiments' / 'protocols' / 'repair_space.json').read_text())
    assert tuple(spec['seeds']) == harness.SEEDS
    assert spec['conditions'] == list(study.CONDITIONS)
    assert spec['training_records'] == 18 and spec['evaluation_records']['replication'] == 60
    assert spec['evaluation_offset'] == 16 and spec['independent_unit'] == 'fixed_record'


def test_checkout_and_installed_paths(monkeypatch, tmp_path):
    monkeypatch.delenv('COAGENT_PROJECT_ROOT', raising=False)
    assert layout.repository_root() == ROOT
    assert layout.default_config().is_file()
    monkeypatch.setenv('COAGENT_DATA_ROOT', str(tmp_path / 'inputs'))
    monkeypatch.setenv('COAGENT_OUTPUT_ROOT', str(tmp_path / 'outputs'))
    assert layout.resolve_legacy_path('Datasets/panel.jsonl') == tmp_path / 'inputs' / 'panel.jsonl'
    assert layout.resolve_legacy_path('Results/run') == tmp_path / 'outputs' / 'run'
    assert layout.resolve_legacy_path(tmp_path / 'explicit.jsonl') == tmp_path / 'explicit.jsonl'


@pytest.mark.parametrize('relative', [
    'datasets/private/record.json', 'datasets/raw/record.jsonl',
    'datasets/unreviewed.jsonl', 'models/cache/config.json',
    'models/unknown.json', 'experiments/outputs/summary.json',
])
def test_release_gate_rejects_private_artifacts(tmp_path, relative):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{}', encoding='utf-8')
    assert not check(tmp_path)['passed']
