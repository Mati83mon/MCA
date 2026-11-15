from __future__ import annotations

import json
from pathlib import Path

import pytest

from mca import MCAConfig, build_offline_orchestrator
from mca.core.evidence_verifier import EvidenceStore
from mca.utils.nli_client import LexicalNLIBackend, NLIClient

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "benchmarks" / "datasets" / "pytania_podchwytliwe_100.json"


@pytest.fixture(scope="session")
def dataset_raw() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def offline_config() -> MCAConfig:
    return MCAConfig(nli_backend="lexical", embedding_backend="hashing")


@pytest.fixture(scope="session")
def store() -> EvidenceStore:
    return EvidenceStore.from_seed()


@pytest.fixture(scope="session")
def nli() -> NLIClient:
    return NLIClient(LexicalNLIBackend())


@pytest.fixture()
def orchestrator(offline_config):
    orch = build_offline_orchestrator(offline_config)
    yield orch
    orch.close()
