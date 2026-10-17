# VisionQA v2 experimental model card

- **Status:** experimental; not promoted
- **Purpose:** retain the 14-class contract while improving WebUI transfer.
- **Architecture:** YOLO11n initialized from VisionQA v1, mixed WebUI plus AtomBlock rehearsal, source-aware partial-label loss.
- **Artifact:** `models/visionqa/visionqa_v2_mixed14/candidate.pt`
- **SHA-256:** `dd86b81d5faf5f018940564ddb344a8ea7caf328b7b567161937e40ab7f7866d`

At the preselected global threshold 0.0625, the untouched 399-screen WebUI confirmation produced precision 0.11309, recall 0.05906, F1 0.07759, and macro-F1 0.10102. AtomBlock mAP50-95 was 0.25541, retaining 91.79% of v1.

The model remains experimental because select had zero true positives, tiny-object recall was 0.0088, mobile recall was 0.0354, gains were source-concentrated, and the lower threshold materially increased prediction/false-positive volume. The production product does not load v2 unless a future milestone explicitly promotes it.
