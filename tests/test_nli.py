"""NLI layer: lexical backend semantics, client caching, label mapping, optional DeBERTa smoke test."""

from __future__ import annotations

import os

import pytest

from mca.telemetry.schemas import NLILabel
from mca.utils.nli_client import LexicalNLIBackend, NLIClient, TransformersNLIBackend


@pytest.mark.parametrize(
    ("premise", "hypothesis", "label"),
    [
        ("Słońce nie jest planetą Układu Słonecznego.", "Słońce jest planetą Układu Słonecznego", NLILabel.CONTRADICTION),
        ("Mars jest czwartą planetą od Słońca.", "Mars jest drugą planetą od Słońca", NLILabel.CONTRADICTION),
        ("Kilogram, symbol kg, jest podstawową jednostką masy w SI.", "Kilogram jest podstawową jednostką długości w SI", NLILabel.CONTRADICTION),
        ("Lista w Pythonie jest modyfikowalna.", "Każda lista w Pythonie jest niemodyfikowalna", NLILabel.CONTRADICTION),
        ("Wenus jest drugą planetą od Słońca.", "Wenus jest drugą planetą od Słońca", NLILabel.ENTAILMENT),
        ("Słońce jest gwiazdą, centralną gwiazdą Układu Słonecznego.", "Słońce jest gwiazdą", NLILabel.ENTAILMENT),
        ("Metr jest jednostką długości.", "Kot jest ssakiem", NLILabel.NEUTRAL),
    ],
)
def test_lexical_backend(premise, hypothesis, label):
    client = NLIClient(LexicalNLIBackend())
    predicted, scores = client.classify(premise, hypothesis)
    assert predicted is label
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_bidirectional_entailment_and_cache():
    client = NLIClient(LexicalNLIBackend())
    assert client.bidirectional_entails("Wenus jest drugą planetą od Słońca", "Drugą planetą od Słońca jest Wenus")
    assert not client.bidirectional_entails("Wenus jest drugą planetą od Słońca", "Mars jest drugą planetą od Słońca")
    before = len(client._cache)
    client.predict([("a b c", "a b c")] * 3)
    assert len(client._cache) == before + 1


def test_transformers_label_mapping():
    mapping = TransformersNLIBackend._resolve_labels({0: "contradiction", 1: "entailment", 2: "neutral"})
    assert mapping == {NLILabel.CONTRADICTION: 0, NLILabel.ENTAILMENT: 1, NLILabel.NEUTRAL: 2}
    mapping = TransformersNLIBackend._resolve_labels({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"})
    assert mapping[NLILabel.ENTAILMENT] == 0
    with pytest.raises(ValueError):
        TransformersNLIBackend._resolve_labels({0: "yes", 1: "no"})


@pytest.mark.skipif(not os.environ.get("MCA_RUN_MODEL_TESTS"), reason="set MCA_RUN_MODEL_TESTS=1 to download DeBERTa")
def test_deberta_cross_encoder():
    pytest.importorskip("transformers")
    backend = TransformersNLIBackend("cross-encoder/nli-deberta-v3-large", device="cpu")
    (contra,) = backend.predict([("The Sun is a star, not a planet.", "The Sun is a planet.")])
    assert max(contra, key=contra.get) is NLILabel.CONTRADICTION
