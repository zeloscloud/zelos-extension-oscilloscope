# Changelog

All notable changes to Zelos Extension Oscilloscope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Needs Zelos 26.0.10 or later.
- The marketplace listing says the oscilloscope is simulated, and search finds it by words such as simulator, signal generator and sine wave.
- Actions are addressed under the extension's name: `Oscilloscope/...` instead of `oscilloscope/...`.

## [0.1.2] - 2026-02-09

### Changed
- A new icon.
- The README describes the extension's real features: burst captures go up to 1 MHz for up to 3 s, and it has no trigger modes or decimation.

## [0.1.1] - 2026-02-09

### Added
- Initial implementation: 1, 2, 4, 8 or 16 simulated channels of sine waves with noise, continuous acquisition, and burst captures at up to 1 MHz for up to 3 s.
- Actions to start and stop acquisition, run a burst capture, set the sample rate, set a channel's frequency, and enable or disable a channel.
