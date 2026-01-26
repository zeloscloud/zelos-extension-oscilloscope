# Zelos Oscilloscope Extension

Multi-channel oscilloscope extension for Zelos with configurable waveform generation, trigger modes, and real-time data streaming.

## Features

- **Multi-channel support**: 1, 2, 4, 8, or 16 channels
- **High sample rates**: Up to 10 MHz (configurable from 1 kHz)
- **Trigger modes**: Continuous, Single-shot, and Repeat (N-shot)
- **Configurable record duration**: From 1 microsecond to 60 seconds
- **Data decimation**: Min-max compression for longer time windows (1x to 128x)
- **Per-channel configuration**: Frequency, amplitude, enable/disable
- **Interactive actions**: Real-time control from Zelos App

## Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `num_channels` | integer | 4 | Number of input channels (1, 2, 4, 8, 16) |
| `sample_rate` | integer | 100000 | Samples per second (1 kHz to 10 MHz) |
| `trigger_mode` | string | continuous | Acquisition mode: continuous, single, repeat |
| `repeat_count` | integer | 10 | Number of captures in repeat mode |
| `record_duration` | number | 0.01 | Duration of each capture window (seconds) |
| `decimation` | integer | 1 | Data compression factor (1, 2, 4, 8, 16, 32, 64, 128) |

### Demo Signal Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ch1_frequency` | number | 50.0 | Channel 1 frequency (Hz) |
| `ch1_amplitude` | number | 5.0 | Channel 1 amplitude (V) |
| `ch2_frequency` | number | 100.0 | Channel 2 frequency (Hz) |
| `ch2_amplitude` | number | 3.3 | Channel 2 amplitude (V) |
| `ch3_frequency` | number | 200.0 | Channel 3 frequency (Hz) |
| `ch3_amplitude` | number | 2.5 | Channel 3 amplitude (V) |
| `ch4_frequency` | number | 500.0 | Channel 4 frequency (Hz) |
| `ch4_amplitude` | number | 1.8 | Channel 4 amplitude (V) |
| `noise_level` | number | 0.05 | Random noise RMS (V) |

## Actions

The extension provides these interactive actions in the Zelos App:

### Acquisition Control
| Action | Description |
|--------|-------------|
| **Run** | Start/resume acquisition |
| **Pause** | Pause acquisition (can be resumed) |
| **Trigger** | Start a single capture (single/repeat modes) |
| **Stop Acquisition** | Stop current acquisition |
| **Get Status** | Get current oscilloscope status |
| **Reset** | Reset all settings to defaults |

### Timing & Trigger
| Action | Description |
|--------|-------------|
| **Set Trigger Mode** | Change between continuous, single, repeat |
| **Set Sample Rate** | Change sample rate (1 kHz to 10 MHz) |
| **Set Record Duration** | Change capture window duration |
| **Set Repeat Count** | Set number of captures in repeat mode |
| **Set Time Base** | Quick presets (1µs, 10µs, 100µs, 1ms, 10ms, 100ms, 1s, 10s) |
| **Set Decimation** | Set data compression factor |

### Channel Configuration
| Action | Description |
|--------|-------------|
| **Configure Channel** | Set channel frequency and amplitude |
| **Set Channel Phase** | Set phase offset for a channel (±360°) |
| **Set Channel Offset** | Set DC offset for a channel (±50V) |
| **Set Waveform Type** | Change waveform type (sine, square, triangle, sawtooth) |
| **Enable Channel** | Enable or disable a channel |
| **Enable All Channels** | Enable all channels at once |
| **Disable All Channels** | Disable all channels at once |
| **Get Channel Info** | Get configuration of a specific channel |
| **Get All Channels** | Get configuration of all channels |
| **Set Noise Level** | Adjust simulated noise level |

## Development

```bash
# Install dependencies
just install

# Run linter and type checks
just check

# Run tests
just test

# Run extension locally
just dev
```

## Links

- [Zelos Documentation](https://docs.zeloscloud.io)
- [SDK Guide](https://docs.zeloscloud.io/sdk)

## License

MIT License
