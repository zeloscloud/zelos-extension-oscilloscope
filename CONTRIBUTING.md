# Contributing

## Prerequisites

- [Zelos](https://zeloscloud.io) 26.0.10 or later
- [Zelos CLI](https://docs.zeloscloud.io/reference/cli/) 0.1.11 or later
- [just](https://github.com/casey/just)
- [uv](https://docs.astral.sh/uv/)

The Zelos guide to [developing extensions](https://docs.zeloscloud.io/latest/sdk/how-to/develop-extensions/) covers the SDK, and [Package and Publish Extensions](https://docs.zeloscloud.io/latest/sdk/how-to/package-extensions/) covers the archive and the marketplace.

## Layout

`extension.toml` describes everything the extension contributes:

| Part | Files | Manifest |
| ---- | ----- | -------- |
| Agent runtime | `main.py`, the Python package, `pyproject.toml`, `config.schema.json` | `[agent]`, `[config]` |

| Path                                        | Purpose                                               |
| ------------------------------------------- | ----------------------------------------------------- |
| `main.py`                                   | Entry point; loads the config and runs the scope      |
| `zelos_extension_oscilloscope/extension.py` | The simulated scope, its trace schema and its actions |
| `tests/test_extension.py`                   | Signal generation, the actions and the burst limits   |

The extension generates its own sine waves with noise, so it needs no hardware.
`demo_signals` in `config.schema.json` sets the frequency and amplitude of channels 1 to 4, and
the noise level.

## Commands

| Command                       | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `just install`                | Install dependencies and the pre-commit hooks        |
| `just format`                 | Format code with ruff                                |
| `just check`                  | Check formatting and lint with ruff                  |
| `just test`                   | Run tests (pytest)                                   |
| `just package`                | Package for the Zelos marketplace                    |
| `just release VERSION`        | Bump version, check, test, commit, tag               |
| `just clean`                  | Remove build output, caches and packages             |
| `just run`                    | Run the extension against `./config.json`            |
| `just ci`                     | Install from the lockfile, check, test and package   |
| `just delete-release VERSION` | Delete a GitHub release and its tag, then force-push |
| `just lock`                   | Update `uv.lock` after changing dependencies         |

`just install` installs the pre-commit hooks in `.pre-commit-config.yaml`. CI runs on every
push and pull request: it checks that `uv.lock` is in sync, then runs `just check` and `just test`.

## Run it locally

`just run` reads `./config.json`. Every setting has a default, so no file is needed for four
channels at 1 kHz:

```bash
just run
```

`just run` connects to the agent at `ZELOS_AGENT_URL`, else `http://localhost:2300`.

## Run it in Zelos

```bash
zelos extensions install-local .
zelos extensions start local.oscilloscope --config '{"num_channels": 4, "sample_rate": 1000}'
zelos extensions logs local.oscilloscope
```

A local install runs from your project directory, so you do not install again after a code
change:

- Agent runtime: stop and start the extension. After a dependency change, stop it and run
  `zelos extensions reinstall local.oscilloscope`.
- `extension.toml`: run `zelos extensions install-local .` again.

## Packaging

```bash
just package
```

The archive lands next to `extension.toml` as `<name>-<version>.tar.gz`, where `<name>` is the
manifest name in lowercase with every other run of characters turned into one hyphen. It holds
`extension.toml`, the README, the icon, the config schema and the paths listed in
`[package] paths`. With an agent runtime it also holds `pyproject.toml`, `uv.lock` and
`actions.json`, the list of actions that run while the extension is stopped.

## Release

Add the release to `CHANGELOG.md` and commit it, then:

```bash
just release 1.0.0
git push --follow-tags
```

The release workflow checks, tests and packages the extension, then attaches the archive to a
GitHub release. The Zelos marketplace picks up each new release.
