---
name: cv-develop
description: Implement or fix computer vision code, model integration, data processing, or streaming pipelines while preserving explicit data and runtime contracts.
---

# Develop a computer vision change

Read applicable AGENTS.md files and inspect the affected implementation, callers, configuration, and tests. Use an existing plan when available; create a short working plan only when complexity warrants it. Complete the requested behavior without bundling unrelated refactors or experiments.

Make affected contracts explicit in the existing code style: color order, layout, dtype, normalization, dimensions, annotation schema, class mapping, and output coordinates. Preserve consistency across training, evaluation, exported models, and inference. When geometric transforms change, carry enough information to map boxes, masks, or keypoints back correctly.

Handle relevant boundary conditions, such as no detections, a final partial batch, variable image sizes, malformed inputs, and unavailable streams. For video changes, preserve timestamp and state semantics; bound buffering and define overload or reconnection behavior when those paths are affected.

For training or model changes, record the dataset/split version, model/checkpoint identifier, configuration, dependencies, and random seeds needed to interpret the result. State remaining nondeterminism. Preserve evaluation separation and compare with a suitable baseline before claiming improved quality.

For accelerated inference or export, check input/output names and shapes, dynamic dimensions, supported operations, precision, and device placement. Use tolerance-based numerical or task-level comparisons appropriate to the change. Avoid unnecessary host/device transfers and unbounded retention of frames or tensors in affected paths.

Run focused checks appropriate to the change. Prefer small synthetic fixtures for contract regressions and representative samples for integration behavior. Do not introduce large model weights or private datasets into Git; use the repository’s artifact mechanism.

Prepare an independent testing handoff containing:

- The intended behavior and acceptance criteria.
- The exact diff target or working-tree state and affected entry points.
- Test commands, fixtures/artifact locations, environment, and resource limits.
- Checks already run and known limitations, clearly identified as author-provided evidence.

Use cv-test when available to obtain the independent pass. That pass must inspect the implementation and derive its own checks. If separate agents are unavailable, state that limitation and report local verification accurately. Address confirmed failures and rerun affected checks before reporting completion.
