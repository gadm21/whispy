# Actuators

An **actuator** turns a `Prediction` into an `Action` — the "A" in the
sense → predict → act loop. Whispy ships three actuator types behind a
single `create_actuator` factory.

## Actuator types

| Type | Action | Use |
|---|---|---|
| `webhook` | POST the prediction to a URL | Zapier, custom endpoints |
| `home_assistant` | Call a Home Assistant service | lights, switches, automations |
| `device` | Send a command back to a device | reconfigure, trigger a sensor |

## Creating an actuator

```python
from whispy import create_actuator

act = create_actuator({
    "type": "webhook",
    "url": "https://example.com/hook",
})
```

## Dispatching an action

Actuators apply **confidence gating** and **label filtering** before
executing — a prediction only fires the action if it clears the configured
threshold and matches the labels you care about.

```python
from whispy.contracts import Prediction

result = act.execute(Prediction(label="occupied", confidence=0.9))
print(result.status)   # ActionStatus
```

`execute` returns an explicit `ActionResult` describing whether the action
was dispatched, gated out, or failed — so the daemon can log and retry
according to the action's `RetryPolicy`.

## In the SMA loop

The `thoth` daemon wires actuators to the active processor: each prediction
is dispatched to the configured actuator, and the `ActionResult` is recorded.
You configure which actuator and which labels/confidence gate it applies in
the deployment or node settings.
