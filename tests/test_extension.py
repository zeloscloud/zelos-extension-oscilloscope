"""Tests for Oscilloscope extension."""

from zelos_extension_oscilloscope import ChannelConfig, Oscilloscope, TriggerMode


def test_oscilloscope_initialization(check) -> None:
    """Test oscilloscope initializes with default config."""
    config = {
        "num_channels": 4,
        "sample_rate": 100000,
        "trigger_mode": "continuous",
    }
    scope = Oscilloscope(config)

    check.that(scope.num_channels, "==", 4)
    check.that(scope.sample_rate, "==", 100000)
    check.that(scope._trigger_mode, "==", TriggerMode.CONTINUOUS)
    check.that(len(scope.channels), "==", 4)


def test_trigger_modes(check) -> None:
    """Test trigger mode enumeration."""
    check.that(TriggerMode.CONTINUOUS.value, "==", "continuous")
    check.that(TriggerMode.SINGLE.value, "==", "single")
    check.that(TriggerMode.REPEAT.value, "==", "repeat")


def test_channel_config(check) -> None:
    """Test channel configuration dataclass."""
    ch = ChannelConfig(frequency=100.0, amplitude=3.3)

    check.that(ch.frequency, "==", 100.0)
    check.that(ch.amplitude, "==", 3.3)
    check.that(ch.phase, "==", 0.0)
    check.that(ch.offset, "==", 0.0)
    check.that(ch.enabled, "is true")


def test_sample_generation(check) -> None:
    """Test waveform sample generation."""
    config = {
        "num_channels": 2,
        "sample_rate": 100000,
        "trigger_mode": "continuous",
        "demo_signals": {
            "ch1_frequency": 1000.0,
            "ch1_amplitude": 5.0,
            "ch2_frequency": 1000.0,
            "ch2_amplitude": 3.3,
            "noise_level": 0.0,  # No noise for deterministic test
        },
    }
    scope = Oscilloscope(config)
    scope.noise_level = 0.0  # Ensure no noise

    # At t=0, sin(0) = 0
    samples = scope._generate_samples(0.0)
    check.that(samples["ch1"], "~=", 0.0, kwargs={"abs": 0.001})
    check.that(samples["ch2"], "~=", 0.0, kwargs={"abs": 0.001})

    # At t=0.00025 (quarter period of 1kHz), sin(pi/2) = 1
    t = 1.0 / (4 * 1000.0)  # Quarter period
    samples = scope._generate_samples(t)
    check.that(samples["ch1"], "~=", 5.0, kwargs={"abs": 0.001})


def test_get_status_action(check) -> None:
    """Test get_status action returns expected fields."""
    config = {
        "num_channels": 4,
        "sample_rate": 100000,
        "trigger_mode": "single",
        "record_duration": 0.01,
    }
    scope = Oscilloscope(config)
    status = scope.get_status()

    check.that(status, "has attribute", "__getitem__")  # Is dict-like
    check.that("running" in status, "is true")
    check.that("trigger_mode" in status, "is true")
    check.that("sample_rate" in status, "is true")
    check.that(status["trigger_mode"], "==", "single")


def test_set_trigger_mode_action(check) -> None:
    """Test set_trigger_mode action."""
    config = {"trigger_mode": "continuous"}
    scope = Oscilloscope(config)

    result = scope.set_trigger_mode("single")

    check.that(result["trigger_mode"], "==", "single")
    check.that(scope._trigger_mode, "==", TriggerMode.SINGLE)


def test_configure_channel_action(check) -> None:
    """Test configure_channel action."""
    config = {"num_channels": 4}
    scope = Oscilloscope(config)

    result = scope.configure_channel(channel=2, frequency=1000.0, amplitude=2.5)

    check.that(result["channel"], "==", 2)
    check.that(scope.channels[1].frequency, "==", 1000.0)
    check.that(scope.channels[1].amplitude, "==", 2.5)


def test_enable_channel_action(check) -> None:
    """Test enable_channel action."""
    config = {"num_channels": 4}
    scope = Oscilloscope(config)

    # Disable channel 1
    result = scope.enable_channel(channel=1, enabled=False)

    check.that(result["enabled"], "is false")
    check.that(scope.channels[0].enabled, "is false")


def test_decimation(check) -> None:
    """Test decimation averaging."""
    config = {"num_channels": 2, "decimation": 4}
    scope = Oscilloscope(config)

    # Create test buffer
    buffer = [
        {"ch1": 1.0, "ch2": 2.0},
        {"ch1": 2.0, "ch2": 4.0},
        {"ch1": 3.0, "ch2": 6.0},
        {"ch1": 4.0, "ch2": 8.0},
    ]

    result = scope._decimate(buffer)

    check.that(result["ch1"], "==", 2.5)  # Average of 1,2,3,4
    check.that(result["ch2"], "==", 5.0)  # Average of 2,4,6,8
