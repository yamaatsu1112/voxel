# AGENTS.md

This file provides guidance to AI agents (Claude Code, Codex, Open Code) when working with code in this repository.

## Project Overview

This is a high-performance voxel engine built in Rust with an Entity-Component-System (ECS) architecture and Vulkan rendering. The engine uses Sparse Voxel Octrees (SVO) for efficient voxel storage and rendering.

## Design Principles

Design the engine with **maintainability and high-performance** as top priorities:
- Prefer zero-copy operations and minimize allocations in hot paths
- Use unsafe code judiciously where performance gains are significant
- Question design choices that sacrifice performance for convenience

## Coding rules
See `docs/coding-rules.md`.

## Other docs
- First list `docs/*` file names
- Read only the files that are relevant to the current task, one by one.

## Project Structure

This project consists of independent crates:

- **engine/**: Cargo workspace containing the engine crates
  - **app**: Core voxel engine library with rendering, systems, and runtime loop
  - **ecs**: Standalone ECS crate shared across the workspace
  - **macro**: Procedural macros for `#[derive(Component)]` and `#[system]`
- **game/**: Example game/application using the engine

### Engine Architecture

**ECS System** (`engine/ecs/src/`):
- **World**: Central registry managing entities, archetypes, and resources
- **Archetype**: Stores entities with identical component sets; uses sparse arrays with free lists for efficient add/remove
- **ArchetypeId**: Hash-based identifier from sorted TypeIds to uniquely identify component combinations
- **Query**: Type-safe iteration over entities matching component signatures
- **System**: Functions that operate on queries and resources, scheduled by the Scheduler
- **Component**: Data attached to entities; derive with `#[derive(Component)]`
- **Resource**: Global singleton data accessed via `Res<T>` or `ResMut<T>`

## Review
When reviewing, always run the app from the game directory. You should check both compilation error and runtime error.
```bash
cd game && cargo run
```

## Common Commands

### Building
```bash
# Build engine workspace
cd engine && cargo build

# Build game
cd game && cargo build

# Build in release mode for performance testing
cd engine && cargo build --release
cd game && cargo build --release
```

### Testing
```bash
# Run all tests (CI-compatible, excludes GPU tests)
cd engine && cargo test --workspace -q
cd game && cargo test --workspace -q

# Run all tests including GPU-dependent tests (local development only)
cd engine && cargo test --workspace -q -- --include-ignored
cd game && cargo test --workspace -q -- --include-ignored

# Run tests for specific crate within engine workspace
cd engine && cargo test -q -p engine-app
```

#### GPU-Dependent Tests
Tests that require a GPU (e.g., Vulkan rendering tests) must be marked with `#[ignore]`:
- GitHub Actions does not have GPU access, so GPU-dependent tests cannot run in CI
- Use `#[ignore]` attribute to exclude these tests from the default `cargo test` run
- Run `cargo test --workspace -- --include-ignored` locally to execute all tests including GPU-dependent ones

```rust
#[test]
#[ignore] // Requires GPU
fn test_vulkan_rendering() {
    // GPU-dependent test code
}
```
