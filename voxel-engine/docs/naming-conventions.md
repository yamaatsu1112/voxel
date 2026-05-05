# Naming Conventions

This document defines naming rules to avoid accidental API naming mistakes.

## Underscore-Prefixed Functions
- Functions with a `_` prefix (e.g., `_foo()`) exist **only** to suppress unused warnings.
- When actually calling such a function, remove the `_` prefix first.
- Never call a function while it still has the `_` prefix.

## Public API Naming
- Do not use a leading underscore (`_`) for any public item (`pub fn`, `pub struct`, `pub mod`, `pub trait`, `pub enum`, `pub const`).
- Leading underscores are allowed only for intentionally unused local variables and private/internal symbols.

## Checklist Before Commit
- Scan for accidental leading underscores in public APIs:
  - `rg -n "pub (fn|struct|enum|trait|mod|const) _" engine game`
- Scan for temporary/debug names that leaked:
  - `rg -n "(tmp|temp|debug|dummy)" engine game`
- If renaming a public API, update all call sites in the same change.

## Review Rule
- When adding new APIs, verify naming consistency against existing modules before merge.
