from __future__ import annotations

from .model import define_model


class Bedrock:
    """Curated Claude-on-Bedrock model definitions.

    Bedrock model IDs differ from the native Anthropic API ids — they are
    AWS-assigned and versioned (e.g. ``anthropic.claude-3-5-sonnet-20241022-v2:0``).
    This class exposes the on-demand, base model ids. For cross-region inference
    profiles (``us.``/``eu.``/``apac.`` prefixes) or models not listed here, build
    one with ``custom_model("bedrock", "<model-id>")``.
    """

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
