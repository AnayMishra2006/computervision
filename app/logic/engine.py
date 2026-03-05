# app/logic/engine.py
#
# Rule-based waste detection engine.
#
# Core rule:
#   IF (person_count == 0) AND (appliance_on == True)
#   AND condition has persisted for WASTE_DELAY_SECONDS
#   THEN → waste_detected = True  →  trigger MQTT alert
#
# The delay avoids false alerts for brief moments of emptiness
# (e.g. a student steps out for 2 seconds to get water).

import time
from app.config import WASTE_DELAY_SECONDS, APPLIANCE_POWER_WATTS, ENERGY_RATE_PER_KWH


class WasteDetector:
    """
    Tracks room state over time and calculates energy waste.

    Parameters
    ----------
    delay_seconds : int
        How many consecutive seconds the room must be empty with appliances ON
        before the waste alert is triggered.
    """

    def __init__(self, delay_seconds: int = WASTE_DELAY_SECONDS):
        self.delay_seconds = delay_seconds

        # Timestamp when the room first became empty with appliances ON
        self._empty_since = None

        # Whether waste is currently being flagged
        self.waste_active = False

        # Cumulative counters (updated every frame the waste condition is active)
        self.total_waste_seconds = 0
        self._last_waste_tick    = None  # used to measure elapsed time between frames

    def update(self, person_count: int, appliance_on: bool) -> bool:
        """
        Feed in the latest detection results.

        Returns True if the waste condition is currently active.
        """
        # Primary waste condition: room is empty AND appliances are running
        condition = (person_count == 0) and appliance_on

        if condition:
            if self._empty_since is None:
                # Start the timer — room just became empty
                self._empty_since = time.time()

            elif time.time() - self._empty_since >= self.delay_seconds:
                # Condition has persisted long enough → trigger alert
                self.waste_active = True

        else:
            # Room is now occupied or appliances are OFF — reset everything
            self._empty_since = None
            self.waste_active = False
            self._last_waste_tick = None

        # Accumulate waste time and calculate energy / cost savings
        if self.waste_active:
            now = time.time()
            if self._last_waste_tick is not None:
                elapsed = now - self._last_waste_tick
                self.total_waste_seconds += elapsed
            self._last_waste_tick = now

        return self.waste_active

    def get_energy_savings(self) -> dict:
        """
        Return a dict of cumulative energy savings since startup.

        Assumption: the system acts on every waste alert and cuts the full
        appliance load for the entire waste duration.  In practice there may be
        a small response delay (ESP32 relay switching ~100 ms) which is
        negligible over minute-scale waste periods.  For a conservative estimate,
        multiply ``energy_saved_kwh`` by an efficiency factor such as 0.9.
        """
        hours_wasted = self.total_waste_seconds / 3600.0
        energy_saved_kwh = (APPLIANCE_POWER_WATTS / 1000.0) * hours_wasted
        cost_saved = energy_saved_kwh * ENERGY_RATE_PER_KWH

        return {
            "total_waste_seconds": int(self.total_waste_seconds),
            "energy_saved_kwh":    round(energy_saved_kwh, 4),
            "cost_saved":          round(cost_saved, 4),
        }