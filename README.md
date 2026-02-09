# Zelos Oscilloscope Extension

Multi-channel oscilloscope extension for [Zelos](https://zeloscloud.io) with configurable waveform generation and real-time data streaming.

## Features

- **Multi-channel**: 1, 2, 4, 8, or 16 channels
- **Continuous acquisition**: Up to 10 kHz sample rate
- **Burst capture**: High-rate capture up to 1 MHz for up to 3 seconds
- **Per-channel control**: Adjustable frequency, amplitude, enable/disable
- **Interactive**: Real-time control from the Zelos App

## Actions

| Action | Description |
|--------|-------------|
| **Start** | Start continuous acquisition |
| **Stop** | Stop acquisition |
| **Burst Capture** | Capture a high-rate burst (up to 1 MHz, up to 3 s) |
| **Set Sample Rate** | Change continuous sample rate (1 - 10,000 Hz) |
| **Set Channel Frequency** | Change a channel's signal frequency |
| **Enable / Disable Channel** | Toggle a channel on or off |

## Configuration

Settings are configured from the Zelos App.

| Setting | Default | Description |
|---------|---------|-------------|
| Number of Channels | 4 | Input channels to acquire (1, 2, 4, 8, or 16) |
| Sample Rate | 1,000 Hz | Continuous acquisition sample rate |

### Demo Signals

Each channel generates a sine wave with configurable frequency and amplitude. Defaults:

| Channel | Frequency | Amplitude |
|---------|-----------|-----------|
| CH1 | 1 Hz | 5.0 V |
| CH2 | 10 Hz | 3.3 V |
| CH3 | 100 Hz | 2.5 V |
| CH4 | 1,000 Hz | 1.8 V |

A noise level (default 0.05 V RMS) is added to all channels.

## Links

- [Zelos Documentation](https://docs.zeloscloud.io)
- [Extension SDK Guide](https://docs.zeloscloud.io/sdk)
