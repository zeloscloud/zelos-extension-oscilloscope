"""Tests for Oscilloscope extension."""

from zelos_extension_oscilloscope import ChannelConfig, Oscilloscope


def test_oscilloscope_initialization(check) -> None:
    """Test oscilloscope initializes with default config."""
    config = {
        "num_channels": 4,
        "sample_rate": 1000,
    }
    scope = Oscilloscope(config)

    check.that(scope.num_channels, "==", 4)
    check.that(scope.sample_rate, "==", 1000)
    check.that(len(scope.channels), "==", 4)


def test_channel_config(check) -> None:
    """Test channel configuration dataclass."""
    ch = ChannelConfig(frequency=1000.0, amplitude=5.0)

    check.that(ch.frequency, "==", 1000.0)
    check.that(ch.amplitude, "==", 5.0)
    check.that(ch.enabled, "is true")


def test_sample_generation(check) -> None:
    """Test waveform sample generation."""
    config = {
        "num_channels": 2,
        "sample_rate": 1000,
        "demo_signals": {
            "ch1_frequency": 1000.0,
            "ch1_amplitude": 5.0,
            "ch2_frequency": 1000.0,
            "ch2_amplitude": 3.3,
            "noise_level": 0.0,
        },
    }
    scope = Oscilloscope(config)
    scope.noise_level = 0.0

    # At t=0, sin(0) = 0
    samples = scope._generate_samples(0.0)
    check.that(samples["ch1"], "~=", 0.0, kwargs={"abs": 0.001})
    check.that(samples["ch2"], "~=", 0.0, kwargs={"abs": 0.001})

    # At t=0.00025 (quarter period of 1kHz), sin(pi/2) = 1
    t = 1.0 / (4 * 1000.0)
    samples = scope._generate_samples(t)
    check.that(samples["ch1"], "~=", 5.0, kwargs={"abs": 0.001})


def test_start_stop_actions(check) -> None:
    """Test start and stop actions."""
    config = {"num_channels": 4}
    scope = Oscilloscope(config)

    result = scope.start_acquisition()
    check.that(result["status"], "==", "running")
    check.that(scope.acquiring, "is true")

    result = scope.stop_acquisition()
    check.that(result["status"], "==", "stopped")
    check.that(scope.acquiring, "is false")


def test_enable_disable_channel_actions(check) -> None:
    """Test enable and disable channel actions."""
    config = {"num_channels": 4}
    scope = Oscilloscope(config)

    result = scope.disable_channel(channel=1)
    check.that(result["enabled"], "is false")
    check.that(scope.channels[0].enabled, "is false")

    result = scope.enable_channel(channel=1)
    check.that(result["enabled"], "is true")
    check.that(scope.channels[0].enabled, "is true")


def test_set_sample_rate_action(check) -> None:
    """Test set sample rate action."""
    config = {"num_channels": 4, "sample_rate": 1000}
    scope = Oscilloscope(config)

    result = scope.set_sample_rate(rate=5000)
    check.that(result["sample_rate"], "==", 5000)
    check.that(scope.sample_rate, "==", 5000)


def test_set_channel_frequency_action(check) -> None:
    """Test set channel frequency action."""
    config = {"num_channels": 4}
    scope = Oscilloscope(config)

    result = scope.set_channel_frequency(channel=2, frequency=500.0)
    check.that(result["channel"], "==", 2)
    check.that(result["frequency"], "==", 500.0)
    check.that(scope.channels[1].frequency, "==", 500.0)
