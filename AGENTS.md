# NInfer — project boundaries

Applicable product and numerical contracts follow. General agent behaviour comes from the unified global rules.

## Current product contract

NInfer is a from-scratch C++/CUDA inference engine for maximum single-GPU inference performance on
a small set of explicitly registered checkpoint artifacts. The supported identities are
`qwen3.6-27b/groupwise-int`, `qwen3.6-27b/nvfp4`, `qwen3.8-27b/groupwise-int`,
`qwen3.8-27b/nvfp4`, and `qwen3.6-35b-a3b/groupwise-int`. The current implementation is compiled
for `sm_120a` and tuned and measured on NVIDIA GeForce RTX 5090. All identities execute Text,
image/video Vision, MTP, prefix reuse, CLI, OpenAI/Anthropic serving, and measurement through the
same public `.ninfer` Engine route; the 35B-A3B target additionally supports text-only DFlash.

The current workload is one GPU and one resident model instance with a startup-fixed one to eight
active requests. The Engine forms one compact decode batch at every round boundary and uses bounded
FIFO ingress with no request preemption. Large-scale or preemptive continuous batching, priority/QoS
scheduling, additional checkpoint targets, and retargeting the implementation to another execution
platform are outside the current product. This is a local, single-owner project. Registered models,
generated artifacts, and the local workflow are trusted.
Requirements derived from a different workload, trust model, or deployment model are out of scope
until that product contract is explicitly changed.

The 27B and 35B-A3B execution packages are peer compile-time Variants of one identity-free Qwen3.6
family runtime. The family owns the shared `SequencePlan<Variant>`, `RequestPlan<Variant>`, and
`Program<Variant>` algorithms; frontend and output semantics; Text/Vision/speculative schedules;
state transactions; workspace composition; and CUDA Graph capture/replay mechanics. Each package
separately owns its registered artifact identities and bindings, immutable model view,
dimensions/storage facts, three closed execution-leaf families, graph frontier data, and Program
instance bytes. No mutable state or device allocation is shared between Programs, neither package
is defined as a delta from the other, and there is no runtime family selection or target-dependent
branch inside family scheduling. All artifacts embed the same six frontend resources, and a
prepared prompt carries no exact-target tag.

## Engineering priorities

Prioritize functional correctness, architectural quality, clear ownership, direct code, and maximum
requested inference performance. Change size, implementation difficulty, short-term simplicity,
and backward compatibility for project-owned contracts are not quality criteria. Low maintenance
cost may distinguish otherwise equivalent designs, but it never justifies worse architecture or
performance. Generality, defensive hardening, formal completeness, broad compatibility, and test
coverage are not goals by themselves.

Prefer explicit target-specific implementation over framework-like abstraction. Do not add generic
model graphs, family base classes, plugin discovery, string-driven execution, hidden device
allocation, runtime weight repacking, or placeholders for hypothetical models or hardware unless an
explicitly changed product contract requires them.

## Sources of truth

Read only current authorities relevant to a live decision in the task. The following list is a
routing map, not a mandatory reading list:

- `README.md` and executable `--help`: delivered capabilities and exact commands;
- `docs/README.md`: public documentation map;
- `docs/cli.md`: CLI input, output, sampling, MTP, and runtime options;
- `docs/serving.md`: OpenAI/Anthropic HTTP behavior;
- `docs/performance.md`: published performance methodology and results;
- `docs/maintainer/concurrent-inference-architecture.md`: bounded ingress, request/slot lifecycle,
  scheduling, batched execution, CUDA Graph, and speculative-concurrency semantics;
- `docs/maintainer/paged-kv-cache.md`: shared KV capacity, page ownership, retention, physical
  layouts, and paged consumer contracts;
- `docs/maintainer/artifact-container.md`, `storage-layouts.md`, and `tensor-formats.md`:
  generic `.ninfer` contracts;
- `docs/maintainer/qwen3.6-27b-artifact.md`, `qwen3.8-27b-artifact.md`, and
  `qwen3.6-35b-a3b-artifact.md`: exact target inventories, conversion, and binding;
- `docs/maintainer/qwen3.6-27b-model.md` and `qwen3.6-35b-a3b-model.md`: exact model mathematics,
  dimensions, and state semantics;
- `docs/maintainer/op-development.md`: Op admission, contracts, implementation ownership,
  qualification, and performance evidence rules;
- `include/ninfer/engine.h` and `include/ninfer/types.h`: in-tree C++ product interface.

Do not survey unrelated references for completeness. Read additional documents only when they
govern a live decision in the current task.

## Product and ownership boundaries

These boundaries govern ordinary implementation work. An explicit architecture task may revise
them, but must update the corresponding active authorities and affected implementation together.

- `.ninfer` is the only C++ product artifact. Do not add `.qus` fallback, extension detection,
  compatibility shims, or a second product lane.
- `include/ninfer/engine.h` and `include/ninfer/types.h` are the opaque Engine interface used by
  in-tree applications and owning host values. NInfer does not currently install or export a C++
  SDK. `include/ninfer/ops/` contains repository-internal semantic Op contracts.
- `src/core` owns device primitives, tensors/views, checked layouts, arenas, graph RAII, physical
  KV-cache containers, and raw transfer mechanisms.
- `src/artifact` owns generic `.ninfer` framing, descriptors, binding primitives, and
  materialization. It has no checkpoint execution semantics.
- `src/ops` owns every semantically closed Op implementation, including fused, fixed-shape, and
  device-specialized paths. Op ownership follows the mathematical or state-transition contract,
  not its first model caller or demonstrated cross-target reuse.
- `src/targets/qwen3_6` owns only the Qwen3.6-family invariants shared by the 27B and 35B-A3B
  targets: tokenizer/template and output semantics, media preprocessing and MRoPE prompt
  construction, owning prepared-prompt/output-session types, semantic weight-view schemas, passive
  Vision definitions, and the fixed
  planning/Program/Text/Vision/speculative/state/workspace/CUDA-Graph algorithms. It has no target
  identity, registry entry, artifact binder, target leaf
  implementation, or storage for a live Program instance.
- `src/targets/<package>` owns its registered checkpoint identities, storage profiles, binder,
  `LoadedModel`, configuration, populated family model-view values and private leaf payloads,
  diagnostics, graph frontier values, and exactly three execution-leaf families: attention
  projection, GDN projection/control, and post-mixer. It aliases and instantiates the family
  runtime types; it does not own a copied Program, Text/Vision/speculative schedule, workspace
  composition, state transaction, or graph-capture algorithm. Leaf Ops remain implemented under
  `src/ops`.
- `src/runtime` owns common contracts, generated-token transaction/publication policy, and the
  public Engine PIMPL. It does not own model mathematics or target state.
- `src/media/decode` consumes already-owned bytes. URL/path/data acquisition belongs to
  `src/product/media_acquire`, CLI, or serving and is not linked into a target.
- `src/product/prompt_input` owns the shared product-side JSON/message-to-owning-input adapter.
- `src/serve` owns protocol translation and transport. CLI, server, and benchmark call only the
  public Engine for inference.
- `tools/convert/<target>`, `tools/reference/<target>`, and `tools/parity/<target>` remain
  target-private conversion, correctness, and diagnostic implementations.

## Compatibility and document lifecycle

Project-owned C++ APIs, CLIs, Python tools, fixtures, reports, formats, and active documentation do
not preserve backward compatibility. When a task replaces project-owned behavior, remove the
obsolete aliases, fallbacks, transition branches, and tests in the affected contract instead of
maintaining two paths. Do not turn that rule into unrelated repository-wide cleanup.

The advertised OpenAI and Anthropic protocol surfaces are real external contracts. A change to
their behavior must update the affected schema tests and serving documentation together.

Integrate stable requirements into the existing active reference. Use a temporary dated plan only
when active work genuinely needs one; a plan is not a substitute for the requested deliverable.
Remove completed or abandoned plans instead of retaining a historical documentation tree. Do not
create parallel `final`, `v2`, or `new-design` references.

## Numerical correctness

When a task changes numerical behavior or makes a numerical claim, identify the mathematical
oracle, represented public inputs, explicit semantic cast/quantization/state boundaries, output
criterion, and real model shapes relevant to that claim. If a route's private precision or
reduction profile matters to the evidence, describe it as an implementation profile rather than a
semantic requirement. Apply exact, tolerance-based, or behavioral comparison according to the
actual semantic contract.

Every floating-point Op has one independent naive FP32/FP64 mathematical oracle; exact transforms
and codecs have one independent exact oracle. The oracle evaluates the complete logical formula
from the represented public inputs and, for packed weights, decodes the signed code with the exact
stored scale. It does not copy a production kernel's staging casts, reduction tree, workspace dtype,
or another implementation's output.

The oracle does not prescribe a production arithmetic path. Unless an intermediate value is an
observable Op output, explicit Cast/quantize/dequantize result, registered codec value, or specified
persistent state, kernels may choose the natural intermediate precision, instruction operands,
reduction association, workspace representation, and kernel decomposition for their route. A fused
kernel is neither required to reproduce an unfused BF16 materialization nor forbidden from using a
lower-precision intermediate when that is the natural qualified implementation. Every production
route is checked directly against the same oracle with a criterion appropriate to its output and
implementation profile; pairwise implementation parity is supplementary evidence only.

Where relevant to the changed behavior, account for numeric-format decode, BF16 fusion order, FP32
GDN state, BF16/INT8 KV, MTP accept/commit state, arena lifetime, and CUDA Graph address stability.
This is a risk map, not a checklist for every numerical task.

## Performance work

Define a performance claim at the level where it matters: operator, schedule, request phase, or
end-to-end inference. Measure that level directly when practical. An isolated microbenchmark can
support an operator-level claim but does not establish an end-to-end improvement.

Use whole-inference profiling when end-to-end attribution remains unresolved. Use kernel profiling
only after a relevant kernel has been identified and a kernel-level answer could materially change
the current design or implementation decision. Do not collect additional profiling data once the
relevant alternatives can be distinguished and the requested claim has adequate support.

Retain concise context sufficient to interpret a reported result: relevant hardware/toolchain,
artifact identity at the descriptive level, workload or command, and summarized measurements. Raw
reports and fixed repository or artifact hashes are not required by default.

## Tests and verification

Add or retain a test only when it protects supported observable behavior or a realistic regression:
numerical kernel/model correctness, `.ninfer` framing/binding, external schema/report behavior, a
small real integration route, GPU lifetime, or a reproduced bug. Do not add tests for coverage,
private file/class shape, getters/constructors, deleted compatibility, source-string scans,
hypothetical failures, or test ceremony.
