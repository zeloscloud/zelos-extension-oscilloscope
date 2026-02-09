"""Oscilloscope extension with multi-channel waveform acquisition."""

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import zelos_sdk

logger = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi

# Burst capture limits
_MAX_BURST_RATE = 1_000_000  # 1 MHz
_MAX_BURST_DURATION = 3.0  # 3 seconds
_BURST_CHUNK_SIZE = 50_000  # samples per chunk for memory-bounded processing


@dataclass(slots=True)
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

        # Pre-cached channel keys to avoid f-string formatting in hot loops
        self._channel_keys = [f"ch{i + 1}" for i in range(self.num_channels)]

        # Timing state
        self._start_time = 0.0
        self._perf_start = 0.0

        # Create trace sources
        self.source = zelos_sdk.TraceSource("oscilloscope")
        self._define_schema()

        logger.info(
            f"Oscilloscope initialized: {self.num_channels} channels, {self.sample_rate} Hz"
        )

    def _define_schema(self) -> None:
        """Define trace events for all channels."""
        fields = [
            zelos_sdk.TraceEventFieldMetadata(key, zelos_sdk.DataType.Float32, "V")
            for key in self._channel_keys
        ]
        # Store event handles to bypass name lookup in hot loops
        self._waveform_event = self.source.add_event("waveform", fields)
        self._burst_event = self.source.add_event("burst", fields)

    def start(self) -> None:
        """Start the oscilloscope."""
        logger.info("Starting oscilloscope")
        self.running = True
        self._start_time = time.time()
        self._perf_start = time.perf_counter()
        self.acquiring = True

    def stop(self) -> None:
        """Stop the oscilloscope."""
        logger.info("Stopping oscilloscope")
        self.running = False
        self.acquiring = False

    def run(self) -> None:
        """Main acquisition loop with drift-compensated timing."""
        _perf_counter = time.perf_counter
        _sleep = time.sleep
        _waveform_log = self._waveform_event.log
        next_sample = _perf_counter()

        while self.running:
            if not self.acquiring:
                _sleep(0.01)
                next_sample = _perf_counter()
                continue

            now = _perf_counter()

            if now < next_sample:
                remaining = next_sample - now
                if remaining > 0.001:
                    _sleep(remaining - 0.0005)
                continue

            # Re-read sample_rate each iteration so set_sample_rate() takes effect
            sample_interval = 1.0 / self.sample_rate

            # If we've fallen too far behind (>10 samples), reset to avoid stale bursts
            if now - next_sample > sample_interval * 10:
                next_sample = now

            t = now - self._perf_start
            _waveform_log(**self._generate_samples(t))

            next_sample += sample_interval

    def _generate_samples(self, t: float) -> dict[str, float]:
        """Generate waveform samples for all channels.

        Args:
            t: Current time in seconds

        Returns:
            Dict mapping channel names to voltage values
        """
        channels = self.channels
        keys = self._channel_keys
        noise_level = self.noise_level
        _sin = math.sin
        samples: dict[str, float] = {}

        for i in range(len(channels)):
            ch = channels[i]
            if not ch.enabled:
                samples[keys[i]] = 0.0
                continue

            value = ch.amplitude * _sin(_TWO_PI * ch.frequency * t + ch.phase) + ch.offset

            if noise_level > 0:
                value += random.gauss(0.0, noise_level)

            samples[keys[i]] = value

        return samples

    # -------------------------------------------------------------------------
    # Actions - Control the oscilloscope from Zelos App
    # -------------------------------------------------------------------------

    @zelos_sdk.action("Start", "Start continuous acquisition")
    def start_acquisition(self) -> dict[str, Any]:
        """Start data acquisition."""
        self.acquiring = True
        self._start_time = time.time()
        self._perf_start = time.perf_counter()
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
        maximum=_MAX_BURST_RATE,
        default=1000000,
        title="Sample Rate (Hz)",
    )
    @zelos_sdk.action.number(
        "duration",
        minimum=0.001,
        maximum=_MAX_BURST_DURATION,
        default=1.0,
        title="Duration (seconds)",
    )
    def burst_capture(self, sample_rate: int, duration: float) -> dict[str, Any]:
        """Capture a high-rate burst of samples.

        Uses NumPy vectorized computation in chunks for maximum throughput.
        Samples are logged with accurate timestamps via log_at().

        Args:
            sample_rate: Samples per second (up to 1 MHz)
            duration: Capture duration in seconds (up to 3 s)
        """
        sample_rate = min(int(sample_rate), _MAX_BURST_RATE)
        duration = min(float(duration), _MAX_BURST_DURATION)
        total_samples = int(sample_rate * duration)
        sample_interval = 1.0 / sample_rate
        sample_interval_ns = int(sample_interval * 1_000_000_000)

        logger.info(f"Burst capture: {sample_rate} Hz for {duration}s ({total_samples} samples)")

        # Pause continuous waveform logging so the receiver isn't flooded
        was_acquiring = self.acquiring
        self.acquiring = False

        # Pre-compute per-channel parameters for enabled channels
        keys = self._channel_keys
        num_channels = len(keys)
        ch_enabled = [ch.enabled for ch in self.channels]
        ch_amplitude = np.array([ch.amplitude for ch in self.channels], dtype=np.float64)
        ch_omega = np.array([_TWO_PI * ch.frequency for ch in self.channels], dtype=np.float64)
        ch_phase = np.array([ch.phase for ch in self.channels], dtype=np.float64)
        ch_offset = np.array([ch.offset for ch in self.channels], dtype=np.float64)
        noise_level = self.noise_level

        # Use direct event handle — bypasses event name string lookup per call
        _burst_log_at = self._burst_event.log_at

        base_time_ns = time.time_ns()
        start_perf = time.perf_counter()

        # Reusable sample dict
        sample = dict.fromkeys(keys, 0.0)

        try:
            # Process in chunks for bounded memory usage
            for chunk_start in range(0, total_samples, _BURST_CHUNK_SIZE):
                chunk_end = min(chunk_start + _BURST_CHUNK_SIZE, total_samples)
                chunk_size = chunk_end - chunk_start

                # Vectorized time array for this chunk
                t = np.arange(chunk_start, chunk_end, dtype=np.float64) * sample_interval

                # Vectorized timestamps (nanoseconds) — convert to Python list to avoid
                # per-element numpy int64 boxing overhead in the inner loop
                ts_list = (
                    base_time_ns
                    + np.arange(chunk_start, chunk_end, dtype=np.int64) * sample_interval_ns
                ).tolist()

                # Pre-compute all channel data vectorized, then convert to Python lists.
                # numpy element access (array[i]) is ~10-50x slower than list access due
                # to object boxing. Bulk .tolist() converts at C speed.
                ch_lists: list[list[float]] = []
                zeros: list[float] | None = None
                for j in range(num_channels):
                    if not ch_enabled[j]:
                        if zeros is None:
                            zeros = [0.0] * chunk_size
                        ch_lists.append(zeros)
                        continue
                    values = ch_amplitude[j] * np.sin(ch_omega[j] * t + ch_phase[j]) + ch_offset[j]
                    if noise_level > 0:
                        values += np.random.normal(0.0, noise_level, chunk_size)
                    ch_lists.append(values.tolist())

                # Log each sample — inner loop uses only Python lists (no numpy access)
                for i in range(chunk_size):
                    for j in range(num_channels):
                        sample[keys[j]] = ch_lists[j][i]
                    _burst_log_at(ts_list[i], **sample)
        finally:
            # Resume continuous waveform logging if it was active before burst
            self.acquiring = was_acquiring

        elapsed = time.perf_counter() - start_perf
        actual_rate = total_samples / elapsed if elapsed > 0 else 0

        logger.info(f"Burst complete: {total_samples} samples in {elapsed:.3f}s")
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
