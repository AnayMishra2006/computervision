# app/logic/engine.py
import time

class WasteDetector:
    """
    Tracks room state over time to avoid false triggers.
    """

    def __init__(self, delay_seconds=5):
        """
            delay_seconds: how long condition must persist
        """
        self.delay_seconds = delay_seconds
        self.empty_since = None
        self.waste_active = False

    def update(self, person_count, appliance_on):
        """
        Update state based on current detection.
        """

        # Condition for potential waste
        condition = person_count == 0 and appliance_on

        if condition:
            # Start timer if not already started
            if self.empty_since is None:
                self.empty_since = time.time()

            # Check if delay threshold reached
            elif time.time() - self.empty_since >= self.delay_seconds:
                self.waste_active = True

        else:
            # Reset if room is occupied or appliance OFF
            self.empty_since = None
            self.waste_active = False

        return self.waste_active