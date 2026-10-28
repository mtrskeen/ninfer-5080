# NInfer RTX 5080 fork

- This fork targets RTX 5080 16 GB (`sm_120a`). Upstream RTX 5090 capabilities and measurements do not establish this fork's supported profile or performance.
- The public inference path uses `.ninfer` artifacts and the Engine interface. Preserve semantic Op ownership in `src/ops` and shared family/runtime algorithms; target-specific bindings and execution leaves do not justify copying the shared schedules.
- For changed numerical behavior, compare each production route directly with an independent mathematical oracle from represented public inputs and explicit cast, quantization and persistent-state boundaries. Kernel staging casts, reduction trees and another kernel's output do not define that oracle. Qualified fused implementations may differ at unobservable intermediates.
- Preserve the affected KV/state precision, speculative accept/commit semantics, allocation lifetimes and CUDA Graph address stability. Apply these checks to the changed path, not every task.
- The advertised OpenAI/Anthropic protocol surfaces are external contracts. Changes to their behavior must update the affected schema tests and serving documentation together.
- Performance evidence names the actual GPU, artifact and workload. An operator microbenchmark supports an operator claim, not an end-to-end speedup; measure the requested level and stop once the alternatives and claim are adequately supported.
