"""
Rule-based logic engine.

Core rule
---------
  IF person_count == 0  AND  appliance_on == True
  THEN waste_detected = True  (after a configurable debounce delay)

The debounce delay prevents nuisance alerts when someone momentarily
leaves the frame (e.g. bending down, standing just outside the FOV).

Energy accounting
-----------------
The engine accumulates how long the waste condition has been active and
converts that into watt-hours saved (once the alert is acted on and the
appliance would be switched off by the ESP32 / smart-plug).
"""
import time
import logging
from app.config import WASTE_DELAY_SECONDS, ROOM_WATTAGE

logger = logging.getLogger(__name__)


class WasteDetector:
    """
    Tracks room state over time to trigger / clear the waste alert.

    Parameters
    ----------
    delay_seconds : int   — seconds the empty+ON condition must persist
    room_wattage  : float — assumed watts for energy-saved calculation
    """

    def __init__(
        self,
        delay_seconds: int = WASTE_DELAY_SECONDS,
        room_wattage:  float = ROOM_WATTAGE,
    ):
        self.delay_seconds = delay_seconds
        self.room_wattage  = room_wattage

        # Time when the empty+ON condition first appeared this cycle
        self._empty_since:  float | None = None

        # Whether waste is currently being flagged
        self._waste_active: bool = False

        # Total seconds the waste condition has been active (cumulative)
        self._waste_seconds: float = 0.0

        # Timestamp of the last update call (for energy accounting)
        self._last_update: float = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, person_count: int, appliance_on: bool) -> bool:
        """
        Evaluate the current frame's state.

        Parameters
        ----------
        person_count : int   — number of people detected this frame
        appliance_on : bool  — whether any appliance is on

        Returns
        -------
        bool — True if waste is currently detected
        """
        now       = time.time()
        elapsed   = now - self._last_update
        self._last_update = now

        # The waste condition: room empty but appliances still running
        condition = (person_count == 0) and appliance_on

        if condition:
            if self._empty_since is None:
                # Start the debounce timer
                self._empty_since = now
                logger.debug("Possible waste — debounce timer started")

            elif now - self._empty_since >= self.delay_seconds:
                # Debounce threshold reached → declare waste
                if not self._waste_active:
                    logger.info(
                        "WASTE DETECTED in room after %ds empty+ON",
                        self.delay_seconds,
                    )
                self._waste_active = True

            # Accumulate energy-waste seconds while alert is active
            if self._waste_active:
                self._waste_seconds += elapsed

        else:
            # Room occupied or appliance already off — reset everything
            if self._waste_active:
                logger.info("Waste condition cleared")
            self._empty_since  = None
            self._waste_active = False

        return self._waste_active

    @property
    def energy_saved_wh(self) -> float:
        """
        Watt-hours that *would* be saved if the appliance is shut off
        every time a waste event was detected (cumulative since startup).
        """
        hours = self._waste_seconds / 3600.0
        return round(self.room_wattage * hours, 4)

    def reset_energy_counter(self):
        """Reset the cumulative energy counter (e.g. at start of each day)."""
        self._waste_seconds = 0.0