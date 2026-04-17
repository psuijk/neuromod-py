import dataclasses

import pytest
from pydantic import BaseModel

from neuromod.tools import Tool, create_tool


class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"


async def get_weather(params: WeatherParams) -> str:
    return f"72°F in {params.location}"


def test_create_tool_basic():
    tool = create_tool(
        name="get_weather",
        description="Get current weather",
        schema=WeatherParams,
        execute=get_weather,
    )
    assert tool.name == "get_weather"
    assert tool.description == "Get current weather"
    assert tool.schema is WeatherParams


def test_create_tool_all_options():
    tool = create_tool(
        name="dangerous",
        description="A dangerous tool",
        schema=WeatherParams,
        execute=get_weather,
        max_calls=5,
        requires_approval=True,
        retry=3,
    )
    assert tool.max_calls == 5
    assert tool.requires_approval is True
    assert tool.retry == 3


def test_create_tool_frozen():
    tool = create_tool(
        name="test",
        description="test",
        schema=WeatherParams,
        execute=get_weather,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        tool.name = "other"  # type: ignore[misc]


def test_tool_schema_is_class():
    tool = create_tool(
        name="test",
        description="test",
        schema=WeatherParams,
        execute=get_weather,
    )
    assert tool.schema is WeatherParams
    assert isinstance(tool.schema, type)


def test_tool_schema_generates_json():
    tool = create_tool(
        name="test",
        description="test",
        schema=WeatherParams,
        execute=get_weather,
    )
    json_schema = tool.schema.model_json_schema()
    assert "properties" in json_schema
    assert "location" in json_schema["properties"]


def test_tool_schema_validates():
    tool = create_tool(
        name="test",
        description="test",
        schema=WeatherParams,
        execute=get_weather,
    )
    parsed = tool.schema.model_validate({"location": "NYC"})
    assert isinstance(parsed, WeatherParams)
    assert parsed.location == "NYC"
    assert parsed.units == "celsius"


def test_create_tool_keyword_only():
    with pytest.raises(TypeError):
        create_tool(  # type: ignore[misc]
            "get_weather",
            "Get weather",
            WeatherParams,
            get_weather,
        )


def test_create_tool_defaults():
    tool = create_tool(
        name="test",
        description="test",
        schema=WeatherParams,
        execute=get_weather,
    )
    assert tool.max_calls is None
    assert tool.requires_approval is False
    assert tool.retry is None
