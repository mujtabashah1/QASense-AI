# VisionQA v1 model card

- **Status:** production / rollback
- **Purpose:** detect 14 QASense UI element classes in screenshots.
- **Architecture:** YOLO11n
- **Artifact:** `models/visionqa/atomblock_yolo11n/best.pt`
- **SHA-256:** `7f527e18271067e2c3ed1c85b557b8b794646b996fed2128bd5a06f8c7234f57`
- **Training data:** AtomBlock-WebUI; 5,420 decodable training images, fixed validation/test splits.

Held-out AtomBlock results: precision 0.47383, recall 0.40132, mAP50 0.37755, and mAP50-95 0.27825. Strong classes include image and structural footer/sidebar/navigation blocks. Radio, checkbox, icon, select, form, and table behavior is materially weaker.

VisionQA identifies UI elements; it is not a visual-defect classifier. Cross-domain transfer is limited, small controls are difficult, and DOM/Vision matches remain evidence rather than guaranteed identity.
