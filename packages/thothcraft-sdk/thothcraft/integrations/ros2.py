"""ROS2 bridge — publish Thoth spatial state and sensor streams as topics.

Requires a ROS2 environment with ``rclpy`` and ``std_msgs`` (e.g. a
Jetson or robot host). Without rclpy the bridge still works in
``dry_run`` mode — useful for development and tests.

    import thothcraft
    from thothcraft.integrations.ros2 import ROS2Bridge

    client = thothcraft.Client.login()
    bridge = ROS2Bridge(client, node_name="thoth")
    bridge.spin()          # publishes /thoth/* until interrupted

Topics published:

    /thoth/spatial_state   std_msgs/String  JSON: all spaces' state
    /thoth/occupancy       std_msgs/String  JSON per space {name, occupied}
    /thoth/people          std_msgs/String  JSON per space {name, count}
    /thoth/chunks          std_msgs/String  raw live-chunk JSON per device
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class _DryPublisher:
    """Collects published messages instead of sending them (no rclpy)."""

    def __init__(self, topic: str):
        self.topic = topic
        self.messages: List[str] = []

    def publish(self, msg: Any) -> None:
        data = getattr(msg, "data", msg)
        self.messages.append(data)
        logger.debug("[dry-run] %s <- %s", self.topic, data)


class ROS2Bridge:
    """Bridge Thoth spatial state + device streams into ROS2 topics.

    Args:
        client:     authenticated ``thothcraft.Client``
        node_name:  ROS2 node name
        poll_s:     seconds between spatial-state publishes
        dry_run:    collect messages in memory instead of publishing
                    (auto-enabled when rclpy is unavailable)
    """

    def __init__(self, client, node_name: str = "thoth",
                 poll_s: float = 2.0, dry_run: bool = False):
        self.client = client
        self.node_name = node_name
        self.poll_s = poll_s
        self._publishers: Dict[str, Any] = {}
        self._node = None
        self._rclpy = None

        if not dry_run:
            try:
                import rclpy  # noqa: F401
                self._rclpy = rclpy
            except ImportError:
                logger.warning("rclpy not available — running in dry_run mode")
                dry_run = True
        self.dry_run = dry_run

        if not dry_run:
            self._rclpy.init()
            self._node = self._rclpy.create_node(node_name)

    # -- publishing ---------------------------------------------------------

    def _publisher(self, topic: str):
        if topic in self._publishers:
            return self._publishers[topic]
        if self.dry_run:
            pub = _DryPublisher(topic)
        else:
            from std_msgs.msg import String
            pub = self._node.create_publisher(String, topic, 10)
        self._publishers[topic] = pub
        return pub

    def publish(self, topic: str, payload: Dict[str, Any]) -> None:
        """Publish a JSON payload on a /thoth/* topic."""
        data = json.dumps(payload, separators=(",", ":"))
        if self.dry_run:
            self._publisher(topic).publish(data)
        else:
            from std_msgs.msg import String
            msg = String()
            msg.data = data
            self._publisher(topic).publish(msg)

    def publish_spatial_state(self) -> List[Dict[str, Any]]:
        """Fetch and publish current spatial state for all spaces."""
        spaces = self.client.spaces_state()
        self.publish("/thoth/spatial_state", {"spaces": spaces})
        for space in spaces:
            name = space.get("name", "unknown")
            self.publish("/thoth/occupancy", {
                "space": name,
                "occupied": bool(space.get("occupied")),
                "confidence": space.get("confidence", 0.0),
            })
            self.publish("/thoth/people", {
                "space": name,
                "count": int(space.get("people_count") or 0),
            })
        return spaces

    def publish_device_chunks(self, device, max_items: Optional[int] = None,
                              on_chunk: Optional[Callable] = None) -> None:
        """Stream one device's live chunks onto /thoth/chunks."""
        for chunk in device.stream():
            self.publish("/thoth/chunks", {
                "device": device.uuid, "chunk": chunk})
            if on_chunk:
                on_chunk(chunk)

    # -- main loop ------------------------------------------------------------

    def spin(self, publish_devices: bool = False) -> None:
        """Publish spatial state every ``poll_s`` until interrupted."""
        try:
            while True:
                self.publish_spatial_state()
                if publish_devices:
                    for device in self.client.devices():
                        if device.online:
                            self.publish_device_chunks(device, max_items=10)
                if self.dry_run:
                    return  # dry_run: single pass for tests
                time.sleep(self.poll_s)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._rclpy is not None:
            try:
                self._rclpy.shutdown()
            except Exception:
                pass
            self._rclpy = None

    def __enter__(self) -> "ROS2Bridge":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()
