# app/mqtt/client.py
#
# Persistent MQTT client for communicating with the ESP32.
# Using a persistent connection (loop_start) is much more efficient than
# paho's publish.single() which opens and closes a TCP socket every time.
#
# Compatible with paho-mqtt >= 2.0.0 (uses CallbackAPIVersion.VERSION2).

import threading
import paho.mqtt.client as mqtt_lib
from app.config import (
    MQTT_BROKER, MQTT_PORT, MQTT_CLIENT_ID,
    MQTT_KEEPALIVE, MQTT_ROOM,
)


class MQTTClient:
    """
    Manages a single, long-lived MQTT connection to the broker.
    Reconnects automatically if the broker goes away.
    Thread-safe: can be called from the CV loop and the API simultaneously.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._connected = False

        # Track the last message sent per topic to avoid spamming the broker
        self._last_messages: dict = {}

        # Create the paho client using the v2 callback API
        self._client = mqtt_lib.Client(
            callback_api_version=mqtt_lib.CallbackAPIVersion.VERSION2,
            client_id=MQTT_CLIENT_ID,
        )

        # Attach callbacks
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Start the connection in a background thread so it never blocks the CV loop
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self):
        """Try to connect to the broker (non-blocking)."""
        try:
            self._client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE)
            # loop_start() spins up a background thread that handles
            # reconnects, pings, and message delivery automatically
            self._client.loop_start()
        except Exception as exc:
            # If the broker is not reachable, we just print a warning.
            # The client will retry when publish() is called later.
            print(f"⚠️  MQTT connect failed ({exc}). Will retry on next publish.")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Called by paho when the connection is established (VERSION2 signature)."""
        if reason_code.is_failure:
            print(f"⚠️  MQTT connection refused: {reason_code}")
        else:
            self._connected = True
            print(f"📡 MQTT connected to {MQTT_BROKER}:{MQTT_PORT}")

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Called by paho when the connection is lost (VERSION2 signature)."""
        self._connected = False
        if reason_code != 0:
            print("⚠️  MQTT unexpectedly disconnected — paho will auto-reconnect.")

    def _publish(self, topic: str, message: str):
        """
        Publish only when the message is different from the last one sent
        on that topic (deduplication).
        """
        with self._lock:
            if self._last_messages.get(topic) == message:
                return  # nothing changed — no need to publish

            if not self._connected:
                # Try to reconnect silently
                self._connect()

            result = self._client.publish(topic, message)
            if result.rc == mqtt_lib.MQTT_ERR_SUCCESS:
                print(f"📡 MQTT → {topic}: {message}")
                self._last_messages[topic] = message
            else:
                print(f"⚠️  MQTT publish failed (rc={result.rc}) for {topic}")

    # ------------------------------------------------------------------
    # Public API used by the rest of the system
    # ------------------------------------------------------------------

    def publish_command(self, topic: str, message: str):
        """
        Publish a device-control command (lights/fan ON|OFF).
        This is the main method called by the logic engine.
        Example topic: wattwatch/room101/lights/cmd
        """
        self._publish(topic, message)

    def publish_waste(self, state: bool):
        """
        Publish the room waste status (for dashboard / other subscribers).
        Topic: wattwatch/<room>/waste  →  "ON" | "OFF"
        """
        topic = f"wattwatch/{MQTT_ROOM}/waste"
        self._publish(topic, "ON" if state else "OFF")

    def disconnect(self):
        """Gracefully stop the MQTT loop and disconnect."""
        self._client.loop_stop()
        self._client.disconnect()