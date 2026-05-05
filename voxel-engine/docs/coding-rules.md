# Coding Rules

## General
- Please implement only the bare minimum required and ensure a minimalist design.
    - For example, don't implement unnecessary getters and setters.
- Do not keep unnecessary functions or code for backward compatibility purposes. When adding new functionality, replace existing implementations.
- Do not implement functions that only call another function. Instead, call the target function directly.
    - For example, if a function `foo()` only calls `bar()`, implement `foo()` as a direct call to `bar()` at the call site, or refactor to eliminate the wrapper.
- Constants must be defined in a single location using imports or similar mechanisms, so that changes in one place propagate to all usages.
    - However, this does not apply between Slang and Rust.

## Rust
- Don't use `mut` if it is not necessary.
    - You can check if it's necessary with `cargo check`.
- Use `pub(crate)` instead of `pub` if the visibility is not required to be public.

### Warnings
- Do not ignore warnings. Fix them appropriately.
- **Unused imports**: Remove the unused import.
- **Methods/functions are never used**:
    - If the method will be needed in the future, add `#[allow(dead_code)]`.
    - If it's unlikely to be needed, remove it.
- **Struct is never constructed**:
    - If the struct will be needed in the future, add `#[allow(dead_code)]`.
    - If it's unlikely to be needed, remove it.
- **Fields are never read**:
    - If the field will be needed in the future, add `#[allow(dead_code)]`.
    - If it's unlikely to be needed, remove it.
- **Do not use items marked with `#[allow(~)]`**. If you want to use them, remove the `#[allow(~)]` attribute.
- **Do not use items with a leading underscore `_`**. If you want to use them, remove the leading underscore.

## Slang

## Testing
- Tests that require GPU access (e.g., Vulkan rendering) must be marked with `#[ignore]`.
    - This allows CI to run without GPU hardware.
    - Use `cargo test --workspace -- --include-ignored` locally to run all tests.

## Comment
- Comments must be written in English.
- Comments should primarily describe functionality, not progress or state changes.
    - Avoid temporal expressions like "is **now**", etc.
