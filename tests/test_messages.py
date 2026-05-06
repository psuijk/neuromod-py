import dataclasses

import pytest

from neuromod.messages import (
    TextContent,
    MediaContent,
    ToolCallContent,
    ToolResultContent,
    Message,
    text,
    image,
    audio,
    document,
    media,
    tool_call,
    tool_result,
    user_message,
    assistant_message,
    system_message,
)


# ── Content creation ───────────────────────────────


def test_text_content_creation():
    tc = TextContent(text="hello")
    assert tc.type == "text"
    assert tc.text == "hello"


def test_media_content_creation():
    mc = MediaContent(data="base64data", mime_type="image/png")
    assert mc.type == "media"
    assert mc.data == "base64data"
    assert mc.mime_type == "image/png"
    assert mc.filename is None

    mc2 = MediaContent(data="base64data", mime_type="application/pdf", filename="doc.pdf")
    assert mc2.filename == "doc.pdf"


def test_tool_call_content_creation():
    tc = ToolCallContent(id="call_1", name="get_weather", arguments={"location": "NYC"})
    assert tc.type == "tool_call"
    assert tc.id == "call_1"
    assert tc.name == "get_weather"
    assert tc.arguments == {"location": "NYC"}


def test_tool_result_content_creation():
    tr = ToolResultContent(call_id="call_1", result="72°F")
    assert tr.type == "tool_result"
    assert tr.call_id == "call_1"
    assert tr.result == "72°F"
    assert tr.name is None
    assert tr.is_error is False

    tr2 = ToolResultContent(call_id="call_1", result="error", name="get_weather", is_error=True)
    assert tr2.name == "get_weather"
    assert tr2.is_error is True


def test_content_frozen():
    tc = TextContent(text="hello")
    with pytest.raises(dataclasses.FrozenInstanceError):
        tc.text = "world"  # type: ignore[misc]

    mc = MediaContent(data="data", mime_type="image/png")
    with pytest.raises(dataclasses.FrozenInstanceError):
        mc.data = "other"  # type: ignore[misc]

    tcc = ToolCallContent(id="1", name="foo", arguments={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        tcc.name = "bar"  # type: ignore[misc]

    trc = ToolResultContent(call_id="1", result="ok")
    with pytest.raises(dataclasses.FrozenInstanceError):
        trc.result = "fail"  # type: ignore[misc]


# ── Message creation ──────────────────────────────


def test_message_creation():
    msg = Message(role="user", content=[TextContent(text="hello")])
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)


# ── Message builders ──────────────────────────────


def test_user_message_from_string():
    msg = user_message("hi")
    assert msg.role == "user"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "hi"


def test_user_message_from_content_list():
    parts = [image("data", "image/png"), text("describe this")]
    msg = user_message(parts)
    assert msg.role == "user"
    assert len(msg.content) == 2
    assert isinstance(msg.content[0], MediaContent)
    assert isinstance(msg.content[1], TextContent)


def test_assistant_message_from_string():
    msg = assistant_message("hello")
    assert msg.role == "assistant"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "hello"


def test_system_message():
    msg = system_message("You are helpful.")
    assert msg.role == "system"
    assert len(msg.content) == 1
    assert isinstance(msg.content[0], TextContent)
    assert msg.content[0].text == "You are helpful."


# ── Extractors ────────────────────────────────────


def test_text_single_part():
    msg = user_message("hello")
    assert msg.text == "hello"


def test_text_multiple_parts():
    msg = Message(role="user", content=[TextContent(text="hello "), TextContent(text="world")])
    assert msg.text == "hello world"


def test_text_no_text_parts():
    msg = Message(role="user", content=[MediaContent(data="data", mime_type="image/png")])
    assert msg.text == ""


def test_text_mixed_content():
    msg = Message(
        role="assistant",
        content=[
            TextContent(text="Here is the result: "),
            ToolCallContent(id="1", name="foo", arguments={}),
            TextContent(text="done"),
        ],
    )
    assert msg.text == "Here is the result: done"


def test_media_property():
    img = MediaContent(data="img", mime_type="image/png")
    doc = MediaContent(data="doc", mime_type="application/pdf")
    msg = Message(role="user", content=[TextContent(text="look"), img, doc])
    assert len(msg.media) == 2
    assert msg.media[0] is img
    assert msg.media[1] is doc


def test_tool_calls_property():
    tc1 = ToolCallContent(id="1", name="foo", arguments={})
    tc2 = ToolCallContent(id="2", name="bar", arguments={"x": 1})
    msg = Message(role="assistant", content=[TextContent(text="calling"), tc1, tc2])
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0] is tc1
    assert msg.tool_calls[1] is tc2


def test_tool_results_property():
    tr1 = ToolResultContent(call_id="1", result="ok")
    tr2 = ToolResultContent(call_id="2", result="error", is_error=True)
    msg = Message(role="user", content=[tr1, tr2])
    assert len(msg.tool_results) == 2
    assert msg.tool_results[0] is tr1
    assert msg.tool_results[1] is tr2


def test_has_tool_calls_true():
    msg = Message(
        role="assistant",
        content=[ToolCallContent(id="1", name="foo", arguments={})],
    )
    assert msg.has_tool_calls is True


def test_has_tool_calls_false():
    msg = Message(role="assistant", content=[TextContent(text="no tools")])
    assert msg.has_tool_calls is False
