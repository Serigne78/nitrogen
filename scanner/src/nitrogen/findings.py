"""What an analyzer reports about a file. Any finding makes the verdict RISK."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    rule: str  # stable identifier, e.g. "safetensors.layout"
    message: str  # human-readable detail
