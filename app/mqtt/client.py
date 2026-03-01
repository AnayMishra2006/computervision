# app/mqtt/client.py
import paho.mqtt.publish as publish

BROKER = "localhost"
PORT = 1883


class MQTTClient:
    def __init__(self):
        # Track last message per topic to prevent spam
        self.last_messages = {}

    def _publish(self, topic, message):
        """Internal publish with duplicate filtering."""
        if self.last_messages.get(topic) != message:
            publish.single(topic, message, hostname=BROKER, port=PORT)
            print(f"📡 MQTT → {topic}: {message}")
            self.last_messages[topic] = message

    def publish_waste(self, room, state):
        """Publish waste status (for dashboard)."""
        topic = f"wattwatch/{room}/waste"
        message = "ON" if state else "OFF"
        self._publish(topic, message)

    def publish_command(self, topic, message):
        """Publish device command (for ESP)."""
        self._publish(topic, message)