"""Live forest-loss inference service (Phase 9).

Serves the Phase 8 carry-forward checkpoint behind a small FastAPI app:
pick a Western Ghats region + two Jan-Apr date windows, the backend pulls the
Sentinel-2 composites from Earth Engine, runs the model, and returns the mask,
cleared hectares and committed aboveground CO2.
"""
