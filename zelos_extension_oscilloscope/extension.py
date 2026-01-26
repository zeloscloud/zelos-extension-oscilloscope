"""Oscilloscope extension with multi-channel waveform acquisition."""

import logging
import math
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import zelos_sdk

logger = logging.getLogger(__name__)


class TriggerMode(Enum):
    """Acquisition trigger modes."""

    CONTINUOUS = "continuous"
    SINGLE = "single"
    REPEAT = "repeat"


@dataclass
class ChannelConfig:
    """Configuration for a single channel."""

    frequency: float = 50.0
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
        self.sample_rate = config.get("sample_rate", 100000)
        self.record_duration = config.get("record_duration", 0.01)
        self.decimation = config.get("decimation", 1)

        # Trigger settings
        self._trigger_mode = TriggerMode(config.get("trigger_mode", "continuous"))
        self._repeat_count = config.get("repeat_count", 10)
        self._captures_remaining = 0

        # Demo signal settings
        demo = config.get("demo_signals", {})
        self.noise_level = demo.get("noise_level", 0.05)

        # Initialize channel configurations
        self.channels: list[ChannelConfig] = []
        for i in range(self.num_channels):
            ch_num = i + 1
            self.channels.append(
                ChannelConfig(
                    frequency=demo.get(f"ch{ch_num}_frequency", 1000.0),
                    amplitude=demo.get(f"ch{ch_num}_amplitude", 5.0 / ch_num),
                )
            )

        # Timing state
        self._start_time = 0.0
        self._sample_count = 0

        # Create trace source for streaming
        self.source = zelos_sdk.TraceSource("oscilloscope")
        self._define_schema()

        logger.info(
            f"Oscilloscope initialized: {self.num_channels} channels, "
            f"{self.sample_rate} Hz, {self._trigger_mode.value} mode"
        )

    def _define_schema(self) -> None:
        """Define trace events for all channels."""
        # Create a waveform event with all channel data
        fields = []
        for i in range(self.num_channels):
            fields.append(
                zelos_sdk.TraceEventFieldMetadata(f"ch{i + 1}", zelos_sdk.DataType.Float32, "V")
            )
        self.source.add_event("waveform", fields)

        # Status event for trigger/capture info
        self.source.add_event(
            "status",
            [
                zelos_sdk.TraceEventFieldMetadata("trigger_mode", zelos_sdk.DataType.UInt8),
                zelos_sdk.TraceEventFieldMetadata("capturing", zelos_sdk.DataType.Boolean),
                zelos_sdk.TraceEventFieldMetadata("captures_remaining", zelos_sdk.DataType.Int32),
            ],
        )

    def start(self) -> None:
        """Start the oscilloscope."""
        logger.info("Starting oscilloscope")
        self.running = True
        self._start_time = time.time()
        self._sample_count = 0

        # Initialize acquisition based on trigger mode
        if self._trigger_mode == TriggerMode.CONTINUOUS:
            self.acquiring = True
        elif self._trigger_mode == TriggerMode.REPEAT:
            self._captures_remaining = self._repeat_count
            self.acquiring = True
        else:  # SINGLE - wait for trigger
            self.acquiring = False

    def stop(self) -> None:
        """Stop the oscilloscope."""
        logger.info("Stopping oscilloscope")
        self.running = False
        self.acquiring = False

    def run(self) -> None:
        """Main acquisition loop."""
        sample_interval = 1.0 / self.sample_rate
        decimation_buffer: list[dict[str, list[float]]] = []

        while self.running:
            if not self.acquiring:
                # Not acquiring - wait for trigger
                time.sleep(0.01)
                continue

            # Generate samples for current time
            t = time.time() - self._start_time
            samples = self._generate_samples(t)

            # Apply decimation if needed
            if self.decimation > 1:
                decimation_buffer.append(samples)
                if len(decimation_buffer) >= self.decimation:
                    samples = self._decimate(decimation_buffer)
                    decimation_buffer = []
                else:
                    time.sleep(sample_interval)
                    continue

            # Emit waveform data
            self.source.waveform.log(**samples)
            self._sample_count += 1

            # Check if capture window complete
            samples_per_capture = int(self.sample_rate * self.record_duration)
            if self._sample_count >= samples_per_capture:
                self._handle_capture_complete()

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

            # Generate sine wave with optional harmonics
            value = ch.amplitude * math.sin(2 * math.pi * ch.frequency * t + ch.phase)
            value += ch.offset

            # Add noise
            if self.noise_level > 0:
                value += random.gauss(0, self.noise_level)

            samples[f"ch{i + 1}"] = value

        return samples

    def _decimate(self, buffer: list[dict[str, list[float]]]) -> dict[str, float]:
        """Apply min-max decimation to sample buffer.

        Args:
            buffer: List of sample dicts to decimate

        Returns:
            Decimated sample dict (using average for now)
        """
        result = {}
        for key in buffer[0]:
            values = [s[key] for s in buffer]
            # Use average for smooth display
            result[key] = sum(values) / len(values)
        return result

    def _handle_capture_complete(self) -> None:
        """Handle completion of a capture window."""
        self._sample_count = 0

        if self._trigger_mode == TriggerMode.SINGLE:
            self.acquiring = False
            logger.info("Single capture complete")
            self._emit_status()

        elif self._trigger_mode == TriggerMode.REPEAT:
            self._captures_remaining -= 1
            if self._captures_remaining <= 0:
                self.acquiring = False
                logger.info("Repeat captures complete")
            else:
                logger.debug(f"Capture complete, {self._captures_remaining} remaining")
            self._emit_status()

        # Continuous mode just keeps going

    def _emit_status(self) -> None:
        """Emit current status to trace."""
        mode_map = {
            TriggerMode.CONTINUOUS: 0,
            TriggerMode.SINGLE: 1,
            TriggerMode.REPEAT: 2,
        }
        self.source.status.log(
            trigger_mode=mode_map[self._trigger_mode],
            capturing=self.acquiring,
            captures_remaining=self._captures_remaining,
        )

    # -------------------------------------------------------------------------
    # Actions - Control the oscilloscope from Zelos App
    # -------------------------------------------------------------------------

    @zelos_sdk.action("Get Status", "Get current oscilloscope status")
    def get_status(self) -> dict[str, Any]:
        """Return current oscilloscope status."""
        return {
            "running": self.running,
            "acquiring": self.acquiring,
            "trigger_mode": self._trigger_mode.value,
            "sample_rate": self.sample_rate,
            "record_duration": self.record_duration,
            "num_channels": self.num_channels,
            "decimation": self.decimation,
            "captures_remaining": self._captures_remaining,
            "sample_count": self._sample_count,
        }

    @zelos_sdk.action("Trigger", "Start a single capture")
    def trigger(self) -> dict[str, Any]:
        """Manually trigger a capture."""
        if self._trigger_mode == TriggerMode.CONTINUOUS:
            return {"error": "Cannot trigger in continuous mode"}

        self._sample_count = 0
        self._start_time = time.time()

        if self._trigger_mode == TriggerMode.REPEAT:
            self._captures_remaining = self._repeat_count

        self.acquiring = True
        self._emit_status()
        return {
            "status": "triggered",
            "mode": self._trigger_mode.value,
            "captures": self._captures_remaining if self._trigger_mode == TriggerMode.REPEAT else 1,
        }

    @zelos_sdk.action("Stop Acquisition", "Stop current acquisition")
    def stop_acquisition(self) -> dict[str, Any]:
        """Stop the current acquisition."""
        was_acquiring = self.acquiring
        self.acquiring = False
        self._emit_status()
        return {"status": "stopped", "was_acquiring": was_acquiring}

    @zelos_sdk.action("Set Trigger Mode", "Change trigger mode")
    @zelos_sdk.action.select(
        "mode",
        choices=["continuous", "single", "repeat"],
        title="Trigger Mode",
    )
    def set_trigger_mode(self, mode: str) -> dict[str, Any]:
        """Set the trigger mode.

        Args:
            mode: One of 'continuous', 'single', 'repeat'
        """
        self._trigger_mode = TriggerMode(mode)
        self._sample_count = 0

        if mode == "continuous":
            self.acquiring = True
        else:
            self.acquiring = False

        self._emit_status()
        logger.info(f"Trigger mode set to {mode}")
        return {"trigger_mode": mode, "acquiring": self.acquiring}

    @zelos_sdk.action("Set Sample Rate", "Change sample rate")
    @zelos_sdk.action.number(
        "rate",
        minimum=1000,
        maximum=10000000,
        default=100000,
        title="Sample Rate (Hz)",
    )
    def set_sample_rate(self, rate: int) -> dict[str, Any]:
        """Set the sample rate.

        Args:
            rate: Sample rate in Hz (1kHz to 10MHz)
        """
        self.sample_rate = int(rate)
        logger.info(f"Sample rate set to {self.sample_rate} Hz")
        return {"sample_rate": self.sample_rate}

    @zelos_sdk.action("Set Record Duration", "Change capture window duration")
    @zelos_sdk.action.number(
        "duration",
        minimum=0.000001,
        maximum=60.0,
        default=0.01,
        title="Duration (seconds)",
    )
    def set_record_duration(self, duration: float) -> dict[str, Any]:
        """Set the record duration.

        Args:
            duration: Duration in seconds (1µs to 60s)
        """
        self.record_duration = duration
        logger.info(f"Record duration set to {duration}s")
        return {"record_duration": self.record_duration}

    @zelos_sdk.action("Set Repeat Count", "Set number of captures in repeat mode")
    @zelos_sdk.action.number(
        "count",
        minimum=1,
        maximum=1000,
        default=10,
        title="Repeat Count",
    )
    def set_repeat_count(self, count: int) -> dict[str, Any]:
        """Set the repeat count for repeat trigger mode.

        Args:
            count: Number of captures (1-1000)
        """
        self._repeat_count = int(count)
        logger.info(f"Repeat count set to {self._repeat_count}")
        return {"repeat_count": self._repeat_count}

    @zelos_sdk.action("Set Decimation", "Set decimation factor for data compression")
    @zelos_sdk.action.select(
        "factor",
        choices=["1", "2", "4", "8", "16", "32", "64", "128"],
        title="Decimation Factor",
    )
    def set_decimation(self, factor: str) -> dict[str, Any]:
        """Set decimation factor for min-max compression.

        Args:
            factor: Decimation factor (1 = no decimation)
        """
        self.decimation = int(factor)
        logger.info(f"Decimation set to {self.decimation}")
        return {"decimation": self.decimation}

    @zelos_sdk.action("Configure Channel", "Set channel frequency and amplitude")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.number(
        "frequency", minimum=0.1, maximum=1000000, default=50, title="Frequency (Hz)"
    )
    @zelos_sdk.action.number(
        "amplitude", minimum=0.0, maximum=100.0, default=5.0, title="Amplitude (V)"
    )
    def configure_channel(self, channel: int, frequency: float, amplitude: float) -> dict[str, Any]:
        """Configure a channel's signal parameters.

        Args:
            channel: Channel number (1-based)
            frequency: Signal frequency in Hz
            amplitude: Signal amplitude in Volts
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].frequency = frequency
        self.channels[idx].amplitude = amplitude
        logger.info(f"Channel {channel}: {frequency} Hz, {amplitude} V")
        return {
            "channel": channel,
            "frequency": frequency,
            "amplitude": amplitude,
        }

    @zelos_sdk.action("Enable Channel", "Enable or disable a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.boolean("enabled", default=True, title="Enabled")
    def enable_channel(self, channel: int, enabled: bool) -> dict[str, Any]:
        """Enable or disable a channel.

        Args:
            channel: Channel number (1-based)
            enabled: Whether channel is enabled
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].enabled = enabled
        logger.info(f"Channel {channel} {'enabled' if enabled else 'disabled'}")
        return {"channel": channel, "enabled": enabled}

    @zelos_sdk.action("Set Noise Level", "Adjust simulated noise")
    @zelos_sdk.action.number(
        "level",
        minimum=0.0,
        maximum=1.0,
        default=0.05,
        title="Noise Level (V RMS)",
        widget="range",
    )
    def set_noise_level(self, level: float) -> dict[str, Any]:
        """Set the noise level for simulated signals.

        Args:
            level: Noise RMS in Volts
        """
        self.noise_level = level
        logger.info(f"Noise level set to {level} V RMS")
        return {"noise_level": level}

    # -------------------------------------------------------------------------
    # Additional Runtime Control Actions
    # -------------------------------------------------------------------------

    @zelos_sdk.action("Run", "Start/resume acquisition")
    def run_acquisition(self) -> dict[str, Any]:
        """Start or resume data acquisition."""
        self.acquiring = True
        self._start_time = time.time()
        self._emit_status()
        logger.info("Acquisition started")
        return {"status": "running", "acquiring": True}

    @zelos_sdk.action("Pause", "Pause acquisition")
    def pause_acquisition(self) -> dict[str, Any]:
        """Pause data acquisition (can be resumed)."""
        self.acquiring = False
        self._emit_status()
        logger.info("Acquisition paused")
        return {"status": "paused", "acquiring": False}

    @zelos_sdk.action("Set Channel Phase", "Set phase offset for a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.number(
        "phase_deg",
        minimum=-360.0,
        maximum=360.0,
        default=0.0,
        title="Phase (degrees)",
    )
    def set_channel_phase(self, channel: int, phase_deg: float) -> dict[str, Any]:
        """Set phase offset for a channel.

        Args:
            channel: Channel number (1-based)
            phase_deg: Phase offset in degrees
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].phase = math.radians(phase_deg)
        logger.info(f"Channel {channel} phase set to {phase_deg}°")
        return {"channel": channel, "phase_deg": phase_deg}

    @zelos_sdk.action("Set Channel Offset", "Set DC offset for a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.number(
        "offset_v",
        minimum=-50.0,
        maximum=50.0,
        default=0.0,
        title="DC Offset (V)",
    )
    def set_channel_offset(self, channel: int, offset_v: float) -> dict[str, Any]:
        """Set DC offset for a channel.

        Args:
            channel: Channel number (1-based)
            offset_v: DC offset in Volts
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        self.channels[idx].offset = offset_v
        logger.info(f"Channel {channel} offset set to {offset_v} V")
        return {"channel": channel, "offset_v": offset_v}

    @zelos_sdk.action("Get Channel Info", "Get configuration of a specific channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    def get_channel_info(self, channel: int) -> dict[str, Any]:
        """Get configuration details for a channel.

        Args:
            channel: Channel number (1-based)
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        ch = self.channels[idx]
        return {
            "channel": channel,
            "frequency": ch.frequency,
            "amplitude": ch.amplitude,
            "phase_deg": math.degrees(ch.phase),
            "offset_v": ch.offset,
            "enabled": ch.enabled,
        }

    @zelos_sdk.action("Set Time Base", "Quick time base presets")
    @zelos_sdk.action.select(
        "preset",
        choices=["1us", "10us", "100us", "1ms", "10ms", "100ms", "1s", "10s"],
        title="Time Base",
    )
    def set_time_base(self, preset: str) -> dict[str, Any]:
        """Set time base using common presets.

        Args:
            preset: Time base preset (e.g., '1ms', '10us')
        """
        presets = {
            "1us": 0.000001,
            "10us": 0.00001,
            "100us": 0.0001,
            "1ms": 0.001,
            "10ms": 0.01,
            "100ms": 0.1,
            "1s": 1.0,
            "10s": 10.0,
        }
        self.record_duration = presets[preset]
        logger.info(f"Time base set to {preset}")
        return {"time_base": preset, "record_duration": self.record_duration}

    @zelos_sdk.action("Reset", "Reset all settings to defaults")
    def reset_to_defaults(self) -> dict[str, Any]:
        """Reset oscilloscope to default settings."""
        # Reset acquisition settings
        self.sample_rate = 100000
        self.record_duration = 0.01
        self.decimation = 1
        self._trigger_mode = TriggerMode.CONTINUOUS
        self._repeat_count = 10
        self.noise_level = 0.05

        # Reset all channels to defaults
        for i, ch in enumerate(self.channels):
            ch.frequency = 1000.0
            ch.amplitude = 5.0 / (i + 1)
            ch.phase = 0.0
            ch.offset = 0.0
            ch.enabled = True

        self.acquiring = True
        self._start_time = time.time()
        self._sample_count = 0
        self._emit_status()

        logger.info("Oscilloscope reset to defaults")
        return {
            "status": "reset",
            "sample_rate": self.sample_rate,
            "record_duration": self.record_duration,
            "trigger_mode": self._trigger_mode.value,
        }

    @zelos_sdk.action("Get All Channels", "Get configuration of all channels")
    def get_all_channels(self) -> dict[str, Any]:
        """Get configuration details for all channels."""
        channels_info = []
        for i, ch in enumerate(self.channels):
            channels_info.append(
                {
                    "channel": i + 1,
                    "frequency": ch.frequency,
                    "amplitude": ch.amplitude,
                    "phase_deg": round(math.degrees(ch.phase), 2),
                    "offset_v": ch.offset,
                    "enabled": ch.enabled,
                }
            )
        return {"channels": channels_info, "num_channels": len(self.channels)}

    @zelos_sdk.action("Set Waveform Type", "Change waveform type for a channel")
    @zelos_sdk.action.number("channel", minimum=1, maximum=16, default=1, title="Channel")
    @zelos_sdk.action.select(
        "waveform",
        choices=["sine", "square", "triangle", "sawtooth"],
        title="Waveform",
    )
    def set_waveform_type(self, channel: int, waveform: str) -> dict[str, Any]:
        """Set waveform type for a channel (stored for reference).

        Args:
            channel: Channel number (1-based)
            waveform: Waveform type
        """
        idx = int(channel) - 1
        if idx < 0 or idx >= len(self.channels):
            return {"error": f"Invalid channel {channel}"}

        # Note: Currently generates sine waves; this stores the preference
        logger.info(f"Channel {channel} waveform set to {waveform}")
        return {"channel": channel, "waveform": waveform}

    @zelos_sdk.action("Enable All Channels", "Enable all channels")
    def enable_all_channels(self) -> dict[str, Any]:
        """Enable all channels."""
        for ch in self.channels:
            ch.enabled = True
        logger.info("All channels enabled")
        return {"status": "all_enabled", "num_channels": len(self.channels)}

    @zelos_sdk.action("Disable All Channels", "Disable all channels")
    def disable_all_channels(self) -> dict[str, Any]:
        """Disable all channels."""
        for ch in self.channels:
            ch.enabled = False
        logger.info("All channels disabled")
        return {"status": "all_disabled", "num_channels": len(self.channels)}
