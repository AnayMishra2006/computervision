"""
MQTT client module.

Publishes device commands and waste-status messages to the MQTT broker
(Mosquitto or similar) which the ESP32 subscribes to.

Topic conventions (unchanged from existing ESP32 firmware):
  wattwatch/{room_id}/lights/cmd   → "ON" | "OFF"
  wattwatch/{room_id}/fan/cmd      → "ON" | "OFF"
  wattwatch/{room_id}/waste        → "ON" | "OFF"

Improvements over the original:
  - Uses paho-mqtt's persistent Client object instead of publish.single()
    so we maintain a persistent TCP connection to the broker.
  - Reconnects automatically if the broker drops the connection.
  - Thread-safe: can be called from the CV thread while FastAPI runs in
    its own thread.
"""
import logging
import threading
import paho.mqtt.client as mqtt_lib
from app.config import MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, ROOM_ID

logger = logging.getLogger(__name__)


class MQTTClient:
    """
    Persistent MQTT client with automatic reconnect.

    The underlying paho Client runs its network loop in a background
    thread (loop_start), so publish() calls are non-blocking.
    """

    def __init__(
        self,
        broker:   str = MQTT_BROKER,
        port:     int = MQTT_PORT,
        username: str = MQTT_USERNAME,
        password: str = MQTT_PASSWORD,
    ):
        self.broker   = broker
        self.port     = port
        self._lock    = threading.Lock()

        # Per-topic deduplication — only send a message when state changes
        self._last_messages: dict[str, str] = {}

        # Set up the paho client
        self._client = mqtt_lib.Client(client_id="wattwatch-cv", clean_session=True)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        if username:
            self._client.username_pw_set(username, password)

        self._connected = False
        self._try_connect()
        self._client.loop_start()   # non-blocking background network thread

    # ------------------------------------------------------------------
    # paho callbacks
    # ------------------------------------------------------------------

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            logger.info("MQTT connected to %s:%d", self.broker, self.port)
        else:
            logger.warning("MQTT connection refused (rc=%d)", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            logger.warning("MQTT unexpected disconnect (rc=%d) — will reconnect", rc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_connect(self):
        """Attempt to connect; log a warning and continue if broker is down."""
        try:
            self._client.connect(self.broker, self.port, keepalive=60)
        except OSError as exc:
            # Broker not reachable — continue running without MQTT
            logger.warning("MQTT broker unreachable (%s:%d): %s", self.broker, self.port, exc)

    def _publish(self, topic: str, message: str):
        """
        Publish *message* to *topic* only if the value has changed.
        Thread-safe via self._lock.
        """
        with self._lock:
            if self._last_messages.get(topic) == message:
                return   # duplicate — skip

            if not self._connected:
                logger.debug("MQTT not connected — skipping publish %s: %s", topic, message)
                self._last_messages[topic] = message   # still track for dedup
                return

            result = self._client.publish(topic, message, qos=1)
            if result.rc == mqtt_lib.MQTT_ERR_SUCCESS:
                logger.info("MQTT → %s: %s", topic, message)
            else:
                logger.warning("MQTT publish failed (rc=%d) for %s", result.rc, topic)

            self._last_messages[topic] = message

    # ------------------------------------------------------------------
    # Public API  (topic names kept identical to original for ESP32 compat)
    # ------------------------------------------------------------------

    def publish_command(self, topic: str, message: str):
        """
        Publish a device command.

        Example:
            mqtt.publish_command("wattwatch/room101/lights/cmd", "OFF")
        """
        self._publish(topic, message)

    def publish_waste(self, room: str, state: bool):
        """
        Publish the waste-detected flag for *room*.

        Example:
            mqtt.publish_waste("room101", True)  →  topic=wattwatch/room101/waste, "ON"
        """
        topic   = f"wattwatch/{room}/waste"
        message = "ON" if state else "OFF"
        self._publish(topic, message)

    def stop(self):
        """Gracefully disconnect and stop the background loop."""
        self._client.loop_stop()
        self._client.disconnect()
