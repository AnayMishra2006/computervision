# app/logic/engine.py
"""
Rule-based waste detection engine.

The Core Rule (from the problem statement)
------------------------------------------
  IF (person_count == 0) AND (appliance_on == True)
  THEN trigger WASTE alert

Debounce / delay
----------------
A single missed detection (YOLO confidence dip for one frame) would
otherwise create spurious alerts and flip relays on/off every second.
We only raise the alert after the waste condition has been *continuously*
true for `delay_seconds`.  This also handles the transient "student
walked out of frame temporarily" case.
"""

import time
from app.config import WASTE_DELAY_SECONDS


class WasteDetector:
    """
    Tracks room state over time and raises a waste alert only after the
    waste condition has persisted for at least *delay_seconds*.

    Attributes
    ----------
    delay_seconds : float  how long the condition must persist before alerting
    empty_since   : float | None  timestamp when empty+appliance condition started
    waste_active  : bool   current alert state
    """

    def __init__(self, delay_seconds: float = WASTE_DELAY_SECONDS):
        self.delay_seconds = delay_seconds
        self.empty_since: float | None = None
        self.waste_active: bool = False

    def update(self, person_count: int, appliance_on: bool) -> bool:
        """
        Feed the latest detection results and return the current waste state.

        Parameters
        ----------
        person_count : int   total people detected (body + face signals merged)
        appliance_on : bool  True when lights / screens appear to be ON

        Returns
        -------
        waste_active : bool  True = WASTE DETECTED alert should fire
        """
        # The waste condition: nobody present but appliances are running
        condition = (person_count == 0) and appliance_on

        if condition:
            if self.empty_since is None:
                # Start the debounce timer on the first frame with this condition
                self.empty_since = time.time()
            elif time.time() - self.empty_since >= self.delay_seconds:
                # Condition has held long enough → raise alert
                self.waste_active = True
        else:
            # Room is occupied OR appliances are already off → reset everything
            self.empty_since = None
            self.waste_active = False

        return self.waste_active