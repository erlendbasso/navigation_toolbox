# AGENTS.md

## Scope

This file applies to the whole repository.

## Project

`navigation_toolbox` is a `no_std` Rust crate for deterministic navigation math.
Keep APIs small, stateless, and suitable for embedded use.

## Development

- Preserve `#![no_std]`.
- Prefer analytic formulas with focused tests against finite differences where practical.
- Keep runtime state, EKF engines, sensor models, timestamp buffering, and coning/sculling accumulators in higher-level crates.
- Avoid new dependencies unless clearly justified.

## Checks

Run before publishing changes:

```sh
cargo fmt
cargo test
cargo clippy --all-targets -- -D warnings
cargo check --target thumbv7em-none-eabihf
cargo doc --no-deps
git diff --check
```
