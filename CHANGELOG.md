# Changelog

## [Phase 1e] - 2026-10-13

### Fixed
- **Trace Sorting Invariant**: Fixed workload trace generation and simulator input validation.
  - **Root Cause Analysis**: `sim.WorkloadGenerator.GenerateTrace` originally iterated through tenant configs and sequentially appended each tenant's requests to a single slice without sorting by arrival time. While `Simulator.Run()` historically sorted input traces internally, custom experiment loops (such as `cmd/exp_heterogeneous/main.go`) iterated directly over the generated slice, assuming it was already chronologically sorted. Consequently, in multi-tenant runs, the second tenant's requests did not enter the queue until all requests of the first tenant had arrived, creating massive artificial queue delays and skewing service shares.
  - **Remediation**:
    1. `GenerateTrace` now enforces the invariant that all returned multi-tenant traces are deterministically sorted by `ArrivalTime()` (with ID tie-breaking).
    2. `Simulator.Run()` now actively validates that input traces are strictly sorted by arrival time, rejecting unsorted input with a panic.
    3. Added unit tests: `TestGeneratedTracesSorted`, `TestSimulatorRejectsUnsortedInput`, and `TestExp1TracePasses`.
  - **Invalidated Results**: The earlier Experiment 2 steady-state numbers (reporting FIFO Chat raw token share as 0.2% / weighted 0.4%, Chat TTFT as 1.3s, and RAG TTFT as 52s) were invalidated artifacts of the unsorted trace. Under corrected arrival sorting, FIFO yields 18.8% raw token share (matching the 18.86% arrival mix) and identical queue wait for both tenants (~17.2s vs ~17.1s).
- **Round-Robin Flow Selection Cursor**:
  - Implemented `RRAdvanceMode` (`RRAdvanceOnPick` vs `RRAdvanceOnDispatch`).
  - Documented that `onPick` models upstream `llm-d-router` behavior (cursor advances on selection, bypassing non-fitting requests under continuous batching), whereas `onDispatch` models idealized 1:1 round-robin (cursor advances only upon confirmed server admission).
