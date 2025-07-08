## Dataset Details

- **Source Domain:** Delhi (565 images)  
- **Target Domain:** Lucknow (598 images)  
  - Active Learning Pool: 65 images  
  - Test Set: 533 images 

## Results

**Class-Agnostic mAP@50 on Target Test Set:**

| No. | Method                                          | mAP@50 |
|-----|--------------------------------------------------|--------|
| 1 | Source-only detector                         | 0.7112 |
| 2 | Active Learning (Entropy Sampling)           | 0.7891 |
| 3 | Active Learning (Margin Sampling)            | 0.7374 |
| 4 | Active Learning (Least Confidence Sampling)  | 0.7814 |
| 5 | Fine-tuned directly on AL pool (no selection)| 0.7586 |