#!/usr/bin/env python3
"""Entry point of the oscilloscope agent runtime."""

import logging
import signal
from types import FrameType

import zelos_sdk
from zelos_sdk.extensions import load_config
from zelos_sdk.hooks.logging import TraceLoggingHandler

from zelos_extension_oscilloscope.extension import Oscilloscope

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Configure logging before SDK initialization
    logging.basicConfig(level=logging.INFO)

    # Initialize the SDK; the name is the trace source name, and actions register under it
    zelos_sdk.init(name="oscilloscope", actions=True)

    # Add trace logging handler to send logs to Zelos
    handler = TraceLoggingHandler("oscilloscope_logger")
    logging.getLogger().addHandler(handler)

    # Load configuration and create the oscilloscope
    config = load_config()
    oscilloscope = Oscilloscope(config)

    # Register interactive actions for the Zelos App
    zelos_sdk.actions_registry.register(oscilloscope)

    def shutdown_handler(signum: int, frame: FrameType | None) -> None:
        """Handle graceful shutdown on SIGTERM or SIGINT."""
        logger.info("Shutting down oscilloscope...")
        oscilloscope.stop()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    logger.info(
        f"Starting oscilloscope: {oscilloscope.num_channels} channels, "
        f"{oscilloscope.sample_rate} Hz"
    )
    oscilloscope.start()
    oscilloscope.run()
