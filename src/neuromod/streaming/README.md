# streaming

Stream event types and the Channel async queue for real-time event consumption.

## Files

- `events.py` — All `StreamEvent` types, `StepResult`, `EventType` enum, `Channel`

## StreamEvent Types

`StreamEvent` is a union of all event types. Each has a `type` field discriminator and a `step_number`.

### Text streaming
| Event | Fields | When |
|-------|--------|------|
| `TextDeltaStreamEvent` | `text, step_number` | A text chunk arrived from the LLM |

### Tool lifecycle
| Event | Fields | When |
|-------|--------|------|
| `ToolCallStartStreamEvent` | `id, name, step_number` | LLM started a tool call |
| `ToolCallDeltaStreamEvent` | `id, arguments_delta, step_number` | Partial tool arguments streaming |
| `ToolCallsReadyStreamEvent` | `calls, step_number` | All tool calls fully parsed |
| `ToolExecutingStreamEvent` | `name, id, step_number` | About to execute a tool |
| `ToolCompleteStreamEvent` | `name, id, result, is_error, duration_ms, step_number` | Tool execution finished |

### Approval
| Event | Fields | When |
|-------|--------|------|
| `ToolApprovalPendingStreamEvent` | `name, id, step_number` | Waiting for approval |
| `ToolApprovalDeniedStreamEvent` | `name, id, step_number` | Approval was denied |

### Step lifecycle
| Event | Fields | When |
|-------|--------|------|
| `StepStartStreamEvent` | `step_number` | A new model loop iteration started |
| `StepCompleteStreamEvent` | `step_number, step: StepResult` | A model loop iteration completed |

## EventType Enum

String enum for matching event types:

```python
from neuromod import EventType

if event.type == EventType.TEXT_DELTA:
    print(event.text)
```

## Channel

An async queue for streaming events between producer and consumer coroutines.

```python
channel = Channel[StreamEvent]()

# Producer side (sync — used as on_event callback)
channel.push(event)    # adds to queue, wakes consumer
channel.close()        # signals no more events

# Consumer side (async)
async for event in channel:
    print(event)
# Loop ends when close() is called
```

Used internally by `Agent.stream()` to bridge the model loop (producer) with the caller (consumer). The `on_event` callback pushes events into the channel synchronously, and the caller consumes them with `async for`.
