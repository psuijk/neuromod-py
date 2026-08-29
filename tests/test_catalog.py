"""Integrity checks over the whole model catalog.

The catalog is hand-maintained reference data, so the failure mode is a typo
— a model filed under the wrong provider, a duplicated id, a limit that got
mangled in an edit. These checks are cheap and catch all three.
"""

from __future__ import annotations

import pytest

from neuromod.models import Bedrock, Claude, Google, Model, Ollama, OpenAI, XAI

CATALOGS = [
    ("Claude", Claude, "anthropic"),
    ("Bedrock", Bedrock, "bedrock"),
    ("Google", Google, "google"),
    ("OpenAI", OpenAI, "openai"),
    ("XAI", XAI, "xai"),
    ("Ollama", Ollama, "ollama"),
]


def models_of(catalog) -> dict[str, Model]:
    return {
        name: value
        for name, value in vars(catalog).items()
        if not name.startswith("_") and isinstance(value, Model)
    }


ALL_ENTRIES = [
    (label, provider, name, model)
    for label, catalog, provider in CATALOGS
    for name, model in models_of(catalog).items()
]

IDS = [f"{label}.{name}" for label, _, name, _ in ALL_ENTRIES]


@pytest.mark.parametrize("label,provider,name,model", ALL_ENTRIES, ids=IDS)
def test_provider_matches_its_catalog(label, provider, name, model):
    assert model.provider == provider


@pytest.mark.parametrize("label,provider,name,model", ALL_ENTRIES, ids=IDS)
def test_limits_are_positive(label, provider, name, model):
    assert model.max_input_tokens > 0
    assert model.max_tokens > 0


@pytest.mark.parametrize("label,provider,name,model", ALL_ENTRIES, ids=IDS)
def test_id_is_clean(label, provider, name, model):
    assert model.id
    assert model.id == model.id.strip()


@pytest.mark.parametrize("label,catalog,provider", CATALOGS, ids=[c[0] for c in CATALOGS])
def test_no_duplicate_ids_within_a_catalog(label, catalog, provider):
    ids = [m.id for m in models_of(catalog).values()]
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, f"{label} defines these ids more than once: {sorted(dupes)}"


def test_bedrock_ids_carry_the_anthropic_prefix():
    """Every Bedrock id must name the anthropic provider, with or without a
    routing prefix — this provider only speaks Anthropic's wire format."""
    for name, model in models_of(Bedrock).items():
        head, _, _ = model.id.rpartition("anthropic.")
        assert "anthropic." in model.id, f"Bedrock.{name} is not an Anthropic model id"
        assert head in ("", "global.", "us.", "eu.", "jp.", "apac."), (
            f"Bedrock.{name} has an unrecognised routing prefix: {head!r}"
        )


def test_catalog_is_not_accidentally_empty():
    for label, catalog, _ in CATALOGS:
        assert models_of(catalog), f"{label} catalog has no models"
