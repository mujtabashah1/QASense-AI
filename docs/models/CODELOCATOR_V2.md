# CodeLocator v2 model card

- **Status:** production / protected
- **Purpose:** rank suspicious Java source lines within natural code regions.
- **Artifact:** `models/codelocator_v2/line_localizer`
- **Model SHA-256:** `9459bfa6b343fc749e5b7653df0f0def005d8781c89388d49466f83e6c0ec288`
- **Training:** CodeBERT with embeddings and encoder layers 0–7 frozen; layers 8–11 and the localization head trained.

Held-out results: Top-1 35.597%, Top-3 64.637%, Top-5 77.986%, MRR 0.53716. Natural-context MRR was 0.53259. The v2 position-only baseline was 0.2826 MRR versus v1's biased 0.8763, showing that the revised task removed the dominant centered-window shortcut.

Scores rank candidates and do not prove the exact buggy line. The evaluated data is Java-specific, inference can be slow on laptop hardware, and the 499 MB model requires Git LFS or external artifact storage.
