"""ROS2 bridge stub for the Robotics track.

Planned shape: a thothcraft ROS2 node that subscribes to the device
sensor stream and publishes perception topics (occupancy, location,
activity) for downstream robot decision-making. Implementation lands
with the Robotics lab track.
"""

from __future__ import annotations


class ROS2Bridge:
    """Placeholder for the Thoth + ROS2 sensor-stream bridge."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ROS2 integration ships with the Robotics lab track. "
            "See thothcraft.integrations.home_assistant for a working bridge."
        )
