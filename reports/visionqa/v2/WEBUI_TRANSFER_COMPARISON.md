# VisionQA v2 WebUI Transfer Comparison

The unchanged 399-screen, 133-group manifest `674ae4c0907b8a1927383d7b0a1a02e285f93cb4ade1ec42becad198b2ec7d45` was evaluated once at confidence 0.25. VisionQA v2 F1 changed from 0.05074 to 0.02659; macro-F1 changed from 0.04835 to 0.05233.

| Class | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|
| block-footer | 0.6667 | 0.0909 | 0.1600 | 0.0660 |
| block-form | 1.0000 | 0.0183 | 0.0359 | 0.0198 |
| block-nav | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| block-sidebar | 0.6667 | 0.0714 | 0.1290 | 0.0792 |
| block-table | 0.3333 | 0.0244 | 0.0455 | 0.0201 |
| button | 0.3696 | 0.0069 | 0.0136 | 0.0066 |
| image | 0.2074 | 0.0654 | 0.0995 | 0.0241 |
| input | 0.4375 | 0.0085 | 0.0166 | 0.0046 |
| link | 0.3077 | 0.0121 | 0.0233 | 0.0141 |
| select | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Responsive class-set Jaccard was 0.3714, with 31 responsive-only miss patterns.
