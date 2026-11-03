---
name: cv-plan
description: Plan a computer vision feature, experiment, or pipeline change using repository evidence, measurable acceptance criteria, and a bounded validation strategy.
---

# Plan a computer vision change

Read applicable AGENTS.md files and inspect the current implementation, configuration, tests, and deployment constraints before proposing changes. Keep this workflow read-only unless the user requests a saved plan or implementation.

Establish the input/output contract: image or video source, annotation schema, coordinate space, color order, tensor layout, dtype, value range, and timing semantics where relevant. Identify which contracts already exist and which the requested change affects.

Define observable success with the user’s operational goal in mind. Choose relevant metrics and evaluation slices: per-class precision/recall, detection or segmentation quality, tracking identity consistency, event-level false alarms, or latency and memory. Record the baseline, intended comparison, and acceptance thresholds when available. Do not invent measured values or business thresholds.

For model or data work, specify dataset and annotation versions, split construction, and possible leakage through adjacent frames, repeated subjects, cameras, or sites. Use an evaluation split that represents the intended deployment scenario. Keep final holdout data out of tuning decisions.

For runtime work, account for decode, preprocessing, inference, postprocessing, queues, and output handling as relevant. Include target hardware, precision, batch size, input resolution, and concurrency in performance comparisons. Distinguish offline throughput from end-to-end streaming latency.

Produce an implementation-ready plan with:

- The goal, relevant existing behavior, and affected paths.
- Proposed changes, dependencies, and meaningful alternatives or tradeoffs.
- Acceptance criteria and the smallest useful validation for each material risk.
- A handoff for an independent testing agent, including commands, fixtures, required artifacts, and resource limits.
- Assumptions and unresolved questions that actually affect implementation.

Resolve routine choices from repository conventions. Ask only for missing decisions that materially change scope or correctness. Do not require training, downloading datasets, or deploying anything merely to write a plan.
