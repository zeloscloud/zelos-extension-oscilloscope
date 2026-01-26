"""Oscilloscope extension with multi-channel waveform acquisition."""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import zelos_sdk

logger = logging.getLogger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""

    frequency: float = 1.0
    amplitude: float = 5.0
    phase: float = 0.0
    offset: float = 0.0
    enabled: bool = True


class Oscilloscope:
    """Multi-channel oscilloscope with configurable waveform generation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize oscilloscope.

        Args:
            config: Configuration from config.json
        """
        self.config = config
        self.running = False
        self.acquiring = False

        # Core settings
        self.num_channels = config.get("num_channels", 4)
        self.sample_rate = config.get("sample_rate", 1000)

        # Demo signal settings
        demo = config.get("demo_signals", {})
        self.noise_level = demo.get("noise_level", 0.05)

        # Initialize channel configurations
        self.channels: list[ChannelConfig] = []
        default_freqs = [1.0, 10.0, 100.0, 1000.0]
        for i in range(self.num_channels):
            ch_num = i + 1
            default_freq = default_freqs[i] if i < len(default_freqs) else 1000.0
            self.channels.append(
                ChannelConfig(
                    frequency=demo.get(f"ch{ch_num}_frequency", default_freq),
                    amplitude=demo.get(f"ch{ch_num}_amplitude", 5.0 / ch_num),
                )
            )

        # Timing state
        self._start_time = 0.0

        # Create trace sources
        self.source = zelos_sdk.TraceSource("oscilloscope")
        self._define_schema()

        logger.info(
            f"Oscilloscope initialized: {self.num_channels} channels, {self.sample_rate} Hz"
        )

    def _define_schema(self) -> None:
        """Define trace events for all channels."""
        # Waveform event for continuous streaming
        fields = []
        for i in range(self.num_channels):
            fields.append(
                zelos_sdk.TraceEventFieldMetadata(f"ch{i + 1}", zelos_sdk.DataType.Float32, "V")
            )
        self.source.add_event("waveform", fields)

        # Burst event for high-rate captures (same schema, different event name)
        self.source.add_event("burst", fields)

    def start(self) -> None:
        """Start the oscilloscope."""
        logger.info("Starting oscilloscope")
        self.running = True
        self._start_time = time.time()
        self.acquiring = True

    def stop(self) -> None:
        """Stop the oscilloscope."""
        logger.info("Stopping oscilloscope")
        self.running = False
        self.acquiring = False

    def run(self) -> None:
        """Main acquisition loop."""
        sample_interval = 1.0 / self.sample_rate

        while self.running:
            if not self.acquiring:
                time.sleep(0.01)
                continue

            # Generate and emit samples
            t = time.time() - self._start_time
            samples = self._generate_samples(t)
            self.source.waveform.log(**samples)

            time.sleep(sample_interval)

    def _generate_samples(self, t: float) -> dict[str, float]:
        """Generate waveform samples for all channels.

        Args:
            t: Current time in seconds

        Returns:
            Dict mapping channel names to voltage values
        """
        samples = {}
        for i, ch in enumerate(self.channels):
            if not ch.enabled:
                samples[f"ch{i + 1}"] = 0.0
                continue

            # Generate sine wave
            value = ch.amplitude * math.sin(2 * math.pi * ch.frequency * t + ch.phase)
            value += ch.offset

            # Add noise
            if self.noise_level > 0:
                value += random.gauss(0, self.noise_level)

            samples[f"ch{i + 1}"] = value

        return samples

    # -------------------------------------------------------------------------
    # Actions - Control the oscilloscope from Zelos App
    # -------------------------------------------------------------------------

    @zelos_sdk.action("Start", "Start continuous acquisition")
    def start_acquisition(self) -> dict[str, Any]:
        """Start data acquisition."""
        self.acquiring = True
        self._start_time = time.time()
        logger.info("Acquisition started")
        return {"status": "running", "sample_rate": self.sample_rate}

    @zelos_sdk.action("Stop", "Stop acquisition")
    def stop_acquisition(self) -> dict[str, Any]:
        """Stop data acquisition."""
        self.acquiring = False
        logger.info("Acquisition stopped")
        return {"status": "stopped"}

    @zelos_sdk.action("Burst Capture", "Capture high-rate burst of samples")
    @zelos_sdk.action.number(
        "sample_rate",
        minimum=1000,
        maximum=10000000,
        default=1000000,
        title="Sample Rate (Hz)",
    )
    @zelos_sdk.action.number(
        "duration",
        minimum=0.001,
        maximum=10.0,
        default=1.0,
        title="Duration (seconds)",
    )
    def burst_capture(self, sample_rate: int, duration: float) -> dict[str, Any]:
        """Capture a high-rate burst of samples.

        Emits data to oscilloscope/burst event at the specified rate.

        Args:
            sample_rate: Samples per second (up to 10 MHz)
            duration: Capture duration in seconds
        """
        sample_rate = int(sample_rate)
        total_samples = int(sample_rate * duration)
        sample_interval = 1.0 / sample_rate

        logger.info(
            f"Starting burst capture: {sample_rate} Hz for {duration}s ({total_samples} samples)"
        )

        start_time = time.time()
        for i in range(total_samples):
            t = i * sample_interval
            samples = self._generate_samples(t)
            self.source.burst.log(**samples)

        elapsed = time.time() - start_time
        actual_rate = total_samples / elapsed if elapsed > 0 else 0

        logger.info(f"Burst capture complete: {total_samples} samples in {elapsed:.3f}s")
        return {
            "status": "complete",
            "samples": total_samples,
            "requested_rate": sample_rate,
            "actual_rate": int(actual_rate),
            "duration": elapsed,
        }

    @zelos_sdk.action("Set Sample Rate", "Change continuous sample rate")
    @zelos_sdk.action.number(
        "rate",
        minimum=1,
        maximum=10000,
        default=1000,
        title="Sample Rate (Hz)",
    )
    def set_sample_rate(self, rate: int) -> dict[str, Any]:
        """Set the continuous acquisition sample rate.

        Args:
            rate: Sample rate in Hz (1 to 10000 for continuous mode)
        """
        self.sample_rate = int(rate)
        logger.info(f"Sample rate set to {self.sample_rate} Hz")
        return {"sample_rate": self.sample_rate}

    @zelos_sdk.action("Set Channel Frequency", "Change a channel's signal frequency")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.number(
        "frequency",
        minimum=0.1,
        maximum=100000,
        default=1.0,
        title="Frequency (Hz)",
    )
    def set_channel_frequency(self, channel: int, frequency: float) -> dict[str, Any]:
        """Set a channel's signal frequency.

        Args:
            channel: Channel number (1-based)
            frequency: Signal frequency in Hz
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].frequency = frequency
        logger.info(f"Channel {channel} frequency set to {frequency} Hz")
        return {"channel": channel, "frequency": frequency}

    @zelos_sdk.action("Enable Channel", "Enable a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    def enable_channel(self, channel: int) -> dict[str, Any]:
        """Enable a channel.

        Args:
            channel: Channel number (1-based)
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].enabled = True
        logger.info(f"Channel {channel} enabled")
        return {"channel": channel, "enabled": True}

    @zelos_sdk.action("Disable Channel", "Disable a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    def disable_channel(self, channel: int) -> dict[str, Any]:
        """Disable a channel.

        Args:
            channel: Channel number (1-based)
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].enabled = False
        logger.info(f"Channel {channel} disabled")
        return {"channel": channel, "enabled": False}
