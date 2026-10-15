"""Portable release checks; fixtures only, no model/API calls."""
import hashlib
import json
import socket
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from Experiment.result_vectors import parse_result_vector
from coagent_repro import replicate_records as harness
from release_tools.check_release import check

ROOT = Path(__file__).resolve().parents[1]


def test_partitions_fixed_and_disjoint():
    rows = harness.partitions(False)
    train = [t for phase, t, _ in rows if phase == 'train']
    test = [t for phase, t, _ in rows if phase == 'eval']
    assert len(train) == 18 and len(test) == 60
    assert not {t.source for t in train} & {t.source for t in test}
    assert len({t.case_id for t in test}) == 60
    assert harness.SEEDS == (101, 137, 173)


def test_smoke_not_a_full_cohort():
    assert len(harness.partitions(True)) == 24
    assert len(harness.partitions(False)) == 78


def test_weight_manifest_has_no_weights():
    manifest = json.loads((ROOT/'src/coagent_repro/weights_manifest.json').read_text())
    assert len(manifest['files']) == 9
    assert sum(f['size'] for f in manifest['files']) == manifest['total_bytes'] == 405398024320
    assert all(len(f['sha256']) == 64 for f in manifest['files'])
    assert not list((ROOT/'src/coagent_repro').glob('*.gguf'))


def test_scheduled_step_required_before_outputs(tmp_path, monkeypatch):
    from argparse import Namespace
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_STEP_ID', raising=False)
    output = tmp_path/'must-not-exist'
    with pytest.raises(RuntimeError, match='compute-node'):
        harness.run(Namespace(output=str(output), mode='smoke', model='qwen2.5:7b', seed=101, port=11434))
    assert not output.exists()


@pytest.mark.parametrize('module', [
    'coagent_repro.replicate_records', 'coagent_repro.strong_records',
    'coagent_repro.v32_records', 'coagent_repro.preflight', 'coagent_repro.v32_preflight',
    'benign_record_feedback.token_probe', 'Experiment.run_co_agent_fuzz',
    'Experiment.run', 'Experiment.compare_fuzzers', 'Experiment.compare_paper_metrics',
    'Experiment.run_validation_matrix', 'Experiment.run_ablation_matrix',
    'Experiment.tune_mutators', 'Experiment.capture_environment',
    'Experiment.audit_validation_artifacts', 'Experiment.audit_ablation_artifacts',
    'co_agent_fuzz.scripts.build_splits', 'co_agent_fuzz.scripts.validate_splits',
    'co_agent_fuzz.scripts.audit_datasets',
])
def test_cli_help_needs_no_model(module):
    result = subprocess.run([sys.executable, '-B', '-m', module, '--help'],
                            cwd=ROOT, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('value', ['[0, 1, 1]', [0, 1, 1], (0, 1, 1), '[False, True, True]'])
def test_record_vectors_preserve_valid_binary_observations(value):
    assert parse_result_vector(value) == [0, 1, 1]


@pytest.mark.parametrize('value', ['__import__("os").system("must-not-run")', '[1, -1]', '[1, 0.5]', '1', '{}'])
def test_result_vectors_do_not_execute_or_fabricate(value):
    with pytest.raises((ValueError, SyntaxError)):
        parse_result_vector(value)


def test_no_real_network_in_tests():
    with socket.socket() as s:
        with pytest.raises(RuntimeError, match='disabled'):
            s.connect(('127.0.0.1', 11434))


def test_source_only_hygiene():
    result = check(ROOT)
    assert result['passed'], result['findings']


def test_packaged_files_are_declared():
    spec = tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert spec['project']['name'] == 'co-agentfuzz'
    assert 'config/*.yaml' in spec['tool']['setuptools']['package-data']['co_agent_fuzz']
    assert 'weights_manifest.json' in spec['tool']['setuptools']['package-data']['coagent_repro']
    assert 'project_identity.json' in spec['tool']['setuptools']['package-data']['Experiment']
    assert 'LICENSE' in spec['tool']['setuptools']['package-data']['PromptFuzz.Fuzzer']
