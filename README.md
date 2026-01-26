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
| `sample_rate` | integer | 1000 | Continuous sample rate in Hz (1 kHz to 10 kHz) |

### Demo Signal Configuration

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `ch1_frequency` | number | 1.0 | Channel 1 frequency (Hz) |
| `ch1_amplitude` | number | 5.0 | Channel 1 amplitude (V) |
| `ch2_frequency` | number | 10.0 | Channel 2 frequency (Hz) |
| `ch2_amplitude` | number | 3.3 | Channel 2 amplitude (V) |
| `ch3_frequency` | number | 100.0 | Channel 3 frequency (Hz) |
| `ch3_amplitude` | number | 2.5 | Channel 3 amplitude (V) |
| `ch4_frequency` | number | 1000.0 | Channel 4 frequency (Hz) |
| `ch4_amplitude` | number | 1.8 | Channel 4 amplitude (V) |
| `noise_level` | number | 0.05 | Random noise RMS (V) |

## Actions

| Action | Description |
|--------|-------------|
| **Start** | Start continuous acquisition |
| **Stop** | Stop acquisition |
| **Burst Capture** | Capture high-rate burst of samples (up to 10 MHz) |
| **Set Sample Rate** | Change continuous sample rate (1-10000 Hz) |
| **Set Channel Frequency** | Change a channel's signal frequency |
| **Enable Channel** | Enable a channel |
| **Disable Channel** | Disable a channel |

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
