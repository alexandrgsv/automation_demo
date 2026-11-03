---
name: cv-test
description: Validate a computer vision change through an independent testing agent that inspects the diff, derives checks, and reports reproducible defects and coverage gaps.
---

# Independently test a computer vision change

## Coordinator

Delegate validation to a separate agent that did not implement the change. Give it the user’s requirements, applicable project instructions, the exact review target, acceptance criteria, available fixtures/artifacts, environment constraints, and resource budget. Identify author-reported checks as unverified evidence; do not supply an expected verdict.

Freeze the validation target: provide a commit, or capture the working-tree diff and untracked files relevant to the change. Do not modify that target while the testing agent is evaluating it. If the implementation changes afterward, request validation of the updated target.

The independent agent may inspect files and run bounded local checks. Default to read-only source review; temporary diagnostic scripts or test artifacts belong outside the source tree. It must not modify implementation files, commit, push, deploy, invoke live write APIs, or delegate again. A request to add persistent tests can explicitly extend its scope to test files only, in a separate worktree or designated paths.

If separate-agent tooling is unavailable, disclose that independent validation could not run. Perform useful local checks, but do not label self-testing as independent testing or claim this workflow passed.

## Testing agent

Read applicable AGENTS.md files. Inspect the complete target diff, relevant surrounding code, callers, and existing tests before choosing checks. For branch comparisons, review changes since the merge base with the intended comparison ref. Consider untracked code when testing uncommitted work.

Derive tests from the requirements and observed failure modes, not by repeating implementation logic. Select relevant cases:

- Image/tensor and annotation contracts: color, normalization, dtype/layout, class IDs, and box/mask/keypoint transforms.
- Empty or boundary inputs, odd image dimensions, partial batches, malformed media, and invalid annotations.
- Video timestamps, ordering, reconnection, state reset, and bounded queues under overload.
- Framework/exported-runtime parity with suitable tolerances, precision differences, and representative shapes.
- Dataset split integrity, evaluation protocol, and meaningful quality slices for model/data changes.
- Latency and memory on the specified hardware when performance is part of acceptance.

Use small synthetic fixtures for deterministic contract checks, then available representative data for integration checks. Run existing relevant tests. Do not start long training, download large artifacts, install new dependencies, or use paid compute unless authorized within the task. Missing GPUs, data, credentials, or dependencies are coverage gaps, not passing results.

Report actionable defects first, ordered P0 (critical), P1 (urgent), P2 (ordinary), P3 (low impact). For each, provide a changed file/line where applicable, the concrete trigger, expected versus actual behavior, and reproduction evidence. Distinguish introduced defects from pre-existing issues and unsupported hypotheses.

Then report the target tested, commands and outcomes, evidence against acceptance criteria, and material coverage gaps. Say “No findings” only when no actionable defects were found; separately state whether validation was complete, partial, or blocked. Never infer a production-quality result from synthetic smoke tests alone.

## Close the loop

The coordinator reviews the evidence and fixes confirmed defects within scope. Revalidate affected behavior after fixes. Report unresolved acceptance failures and unavailable checks explicitly rather than presenting an incomplete pass as success.
