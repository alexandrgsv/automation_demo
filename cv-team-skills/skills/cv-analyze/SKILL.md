---
name: cv-analyze
description: Investigate a computer vision failure, quality regression, performance bottleneck, or data issue and report evidence-backed causes and next actions.
---

# Analyze a computer vision system

Read applicable AGENTS.md files. Establish the reported symptom, affected inputs, expected behavior, and exact code/configuration/model/data versions available. Inspect relevant call paths and tests. Keep the investigation read-only unless fixes or generated diagnostic artifacts are requested.

Trace the affected path from source and annotations through preprocessing, model output, postprocessing, and the consumer. Check relevant contracts rather than assuming common defaults:

- RGB/BGR, HWC/CHW, normalization, dtype, and batch dimensions.
- Resize, crop, padding, and inverse coordinate transforms; xyxy/xywh, normalized/pixel coordinates, and mask interpolation.
- Class indices, label mappings, confidence thresholds, NMS, and empty outputs.
- For video: timestamps, frame sampling, dropped frames, ordering, stream reconnection, and tracker state across stream boundaries.

Choose diagnostics that discriminate between explanations. Reproduce on a small representative fixture when possible, inspect intermediate values or overlays, and compare with a known baseline. Separate dataset defects, model limitations, implementation defects, and environmental differences using observable evidence.

For quality comparisons, use the same evaluation protocol and inspect relevant class, camera, site, lighting, occlusion, and object-size slices. Check leakage and annotation consistency before attributing gains to a model change.

For performance investigations, measure relevant stages on representative hardware. Include warmup and synchronization when needed for asynchronous accelerators; report latency distribution, throughput, and memory under stated load. Check queue growth and backpressure for streaming workloads.

Report findings in order of impact, with the affected path and supporting evidence. Separate confirmed causes from hypotheses; attach a discriminating next check to unresolved hypotheses. State what could not be reproduced and why. Recommend the smallest justified fix or experiment, without silently changing thresholds or evaluation data to improve results.
