from __future__ import annotations

from .model import define_model


class Bedrock:
    """Claude-on-Bedrock model definitions.

    This provider speaks Anthropic's Messages wire format over Bedrock's
    ``InvokeModel`` endpoint, so only Claude models work here. Nova, Llama,
    Mistral, Titan and the rest of the Bedrock catalog use different request
    shapes and would need their own provider.

    Bedrock has two families of Claude model id, and both are reachable
    through ``InvokeModel``:

    **Unversioned ids** (Claude Opus 4.7 and later). Served by the same
    infrastructure as Bedrock's native Messages endpoint. Pass them as-is —
    no inference-profile prefix.

    **ARN-versioned ids** (Claude Opus 4.6 and earlier). AWS serves these
    through cross-region inference only: passing the bare base id fails with
    an HTTP 400 telling you to use an inference profile. The entries below
    therefore carry the ``global.`` prefix, which routes dynamically with no
    pricing premium. For data residency, swap it for a regional prefix
    (``us.``/``eu.``/``jp.``/``apac.``, a 10% premium) via ``custom_model``::

        custom_model("bedrock", "us.anthropic.claude-opus-4-6-v1",
                     max_input=1_000_000, max_tokens=128_000)

    Region is required and resolved from ``BEDROCK_REGION`` / ``AWS_REGION``.
    """

    # ── Unversioned ids (Opus 4.7+) — pass directly, no profile prefix ──
    Fable5 = define_model("bedrock", "anthropic.claude-fable-5", max_input=1_000_000, max_tokens=128_000)
    Opus5 = define_model("bedrock", "anthropic.claude-opus-5", max_input=1_000_000, max_tokens=128_000)
    Sonnet5 = define_model("bedrock", "anthropic.claude-sonnet-5", max_input=1_000_000, max_tokens=128_000)
    Opus4_8 = define_model("bedrock", "anthropic.claude-opus-4-8", max_input=1_000_000, max_tokens=128_000)
    Opus4_7 = define_model("bedrock", "anthropic.claude-opus-4-7", max_input=1_000_000, max_tokens=128_000)
    Haiku4_5 = define_model("bedrock", "anthropic.claude-haiku-4-5", max_input=200_000, max_tokens=64_000)

    # ── ARN-versioned ids (Opus 4.6 and earlier) — require an inference profile ──
    Opus4_6 = define_model(
        "bedrock", "global.anthropic.claude-opus-4-6-v1",
        max_input=1_000_000, max_tokens=128_000,
    )
    Sonnet4_6 = define_model(
        "bedrock", "global.anthropic.claude-sonnet-4-6",
        max_input=1_000_000, max_tokens=128_000,
    )
    Opus4_5 = define_model(
        "bedrock", "global.anthropic.claude-opus-4-5-20251101-v1:0",
        max_input=200_000, max_tokens=64_000,
    )
    Sonnet4_5 = define_model(
        "bedrock", "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        max_input=200_000, max_tokens=64_000,
    )

    # ── Pre-existing entries ────────────────────────
    # These predate the catalog refresh and are kept for backwards
    # compatibility. Claude 3.5 Sonnet v2 / 3.5 Sonnet / 3 Opus / 3 Sonnet no
    # longer appear in AWS's supported-model list or Anthropic's Bedrock
    # tables, and Claude 3.5 Haiku is marked deprecated — they are likely
    # retired. They also carry bare base ids, which cross-region-inference
    # models reject; if one still resolves for you it will need a ``us.``
    # prefix. See the note in models/README.md.
    Claude3_5_Sonnet_v2 = define_model(
        "bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_input=200_000, max_tokens=8_192,
    )
    Claude3_5_Sonnet = define_model(
        "bedrock", "anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_input=200_000, max_tokens=8_192,
    )
    Claude3_5_Haiku = define_model(
        "bedrock", "anthropic.claude-3-5-haiku-20241022-v1:0",
        max_input=200_000, max_tokens=8_192,
    )
    Claude3_Opus = define_model(
        "bedrock", "anthropic.claude-3-opus-20240229-v1:0",
        max_input=200_000, max_tokens=4_096,
    )
    Claude3_Sonnet = define_model(
        "bedrock", "anthropic.claude-3-sonnet-20240229-v1:0",
        max_input=200_000, max_tokens=4_096,
    )
    Claude3_Haiku = define_model(
        "bedrock", "anthropic.claude-3-haiku-20240307-v1:0",
        max_input=200_000, max_tokens=4_096,
    )
