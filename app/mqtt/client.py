# app/mqtt/client.py
"""
MQTT client for communicating with the ESP32 microcontroller.

Architecture
------------
The ESP32 subscribes to the topics below and acts on commands:
  wattwatch/room101/lights/cmd  →  "ON" / "OFF"
  wattwatch/room101/fan/cmd     →  "ON" / "OFF"
  wattwatch/room101/waste       →  "ON" / "OFF"

We use *paho-mqtt's* fire-and-forget `publish.single()` helper rather
than maintaining a persistent connection, which keeps the code simple
and avoids having a background MQTT thread that could interfere with
the CV loop.

Error handling
--------------
If the broker is unreachable (e.g., running the demo without a local
MQTT broker) we catch the exception, print a warning, and continue.
The CV pipeline must NEVER crash because of a missing broker.
"""

import paho.mqtt.publish as publish
from app.config import MQTT_BROKER, MQTT_PORT


class MQTTClient:
    """
    Thin wrapper around paho-mqtt's single-shot publisher.

    Keeps a cache of the last value sent per topic so that we only
    transmit when the state actually *changes*, preventing the ESP32
    from being flooded with identical messages every frame.
    """

    def __init__(self):
        # topic → last message sent (used for deduplication)
        self._last: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _publish(self, topic: str, message: str) -> None:
        """
        Publish *message* to *topic*, but only if the value changed.
        Silently swallows broker-unavailable errors so the CV loop keeps
        running during demos where no broker is running.
        """
        if self._last.get(topic) == message:
            return  # nothing changed – skip network round-trip

        try:
            publish.single(topic, message, hostname=MQTT_BROKER, port=MQTT_PORT)
            print(f"📡 MQTT → {topic}: {message}")
            self._last[topic] = message
        except Exception as exc:
            # Broker unreachable, network error, etc.
            print(f"⚠️  MQTT publish failed ({topic}): {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def publish_waste(self, room: str, state: bool) -> None:
        """
        Broadcast the current waste status for a room.
        Consumed by the dashboard / ESP32 alert LED.
        """
        topic = f"wattwatch/{room}/waste"
        self._publish(topic, "ON" if state else "OFF")

    def publish_command(self, topic: str, message: str) -> None:
        """
        Send a device command directly (lights, fan, etc.) to the ESP32.
        Called by the main loop whenever the waste state changes.
        """
        self._publish(topic, message)