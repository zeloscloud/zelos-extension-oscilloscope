#!/usr/bin/env python3
"""Oscilloscope extension - Multi-channel waveform acquisition for Zelos."""

import logging
import signal
from types import FrameType

import zelos_sdk
from zelos_sdk.extensions import load_config
from zelos_sdk.hooks.logging import TraceLoggingHandler

from zelos_extension_oscilloscope.extension import Oscilloscope

# Configure basic logging before SDK initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from config.json (with schema defaults applied)
config = load_config()

# Create oscilloscope
oscilloscope = Oscilloscope(config)

# Register actions BEFORE SDK init
zelos_sdk.actions_registry.register(oscilloscope)

# Initialize SDK
zelos_sdk.init(name="oscilloscope", actions=True)

# Add trace logging handler to send logs to Zelos
handler = TraceLoggingHandler("oscilloscope_logger")
logging.getLogger().addHandler(handler)


def shutdown_handler(signum: int, frame: FrameType | None) -> None:
    """Handle graceful shutdown on SIGTERM or SIGINT.

    Args:
        signum: Signal number (SIGTERM=15, SIGINT=2)
        frame: Current stack frame
    """
    logger.info("Shutting down oscilloscope...")
    oscilloscope.stop()


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# Run
if __name__ == "__main__":
    logger.info(
        f"Starting oscilloscope: {oscilloscope.num_channels} channels, "
        f"{oscilloscope.sample_rate} Hz"
    )
    oscilloscope.start()
    oscilloscope.run()
