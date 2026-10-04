# D1 leave-one-subject-out summary (8 held-out subjects: 0, 1, 2, 4, 5, 6, 7, 8)

Top-1 word accuracy (%), mean ± sd over held-out subjects; every decoder on traces resampled to 50 Hz.
`aug > clean` = held-out subjects where the augmented model beats the clean one.

| Condition | SHARK2+smooth@50Hz | clean@50Hz | aug@50Hz | aug − clean | aug > clean |
|---|---|---|---|---|---|
| clean (50 Hz) | 74.5 ± 1.4 | 90.5 ± 3.2 | 91.8 ± 2.6 | +1.3 | 4/8 |
| 30 fps | 77.4 ± 1.1 | 89.8 ± 3.6 | 91.7 ± 3.2 | +1.9 | 6/8 |
| 12 fps | 78.8 ± 2.5 | 86.8 ± 6.4 | 90.4 ± 4.6 | +3.6 | 8/8 |
| jitter 0.1 | 73.2 ± 1.8 | 90.8 ± 2.9 | 92.0 ± 2.8 | +1.3 | 4/8 |
| jitter 0.2 | 69.2 ± 2.3 | 90.4 ± 3.0 | 92.0 ± 2.8 | +1.6 | 4/8 |
| jitter 0.3 | 63.5 ± 2.5 | 89.7 ± 2.9 | 91.3 ± 2.9 | +1.6 | 6/8 |
| webcam: 12 fps + 0.15 + 100 ms | 70.6 ± 8.4 | 75.6 ± 13.0 | 83.4 ± 10.6 | +7.8 | 8/8 |
| harsh: 10 fps + 0.3 + 100 ms | 62.3 ± 7.7 | 70.5 ± 13.8 | 79.0 ± 12.0 | +8.5 | 8/8 |

Per held-out subject, webcam preset (12 fps + 0.15 + 100 ms), top-1 (%):

| Subject | n | SHARK2+smooth@50Hz | clean@50Hz | aug@50Hz |
|---|---|---|---|---|
| 0 | 1845 | 73.3 | 89.1 | 92.8 |
| 1 | 2840 | 73.0 | 75.9 | 87.5 |
| 2 | 2447 | 70.5 | 70.9 | 79.0 |
| 4 | 2631 | 69.8 | 78.9 | 83.5 |
| 5 | 2803 | 50.9 | 46.4 | 59.3 |
| 6 | 1178 | 77.8 | 77.8 | 89.5 |
| 7 | 2223 | 76.9 | 83.9 | 88.9 |
| 8 | 1007 | 72.2 | 81.6 | 86.4 |

Word-frequency prior (score = log-lik + λ·log p(word)), top-1 (%), mean over 8 subjects.
SHARK2's λ is tuned per held-out subject on the other subjects (chosen: 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5).

| Condition | Decoder | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | SHARK2 tuned λ |
|---|---|---|---|---|---|---|---|
| clean (50 Hz) | SHARK2+smooth | 74.8 ± 1.8 | 82.9 ± 1.6 | 84.8 ± 1.3 | 84.6 ± 1.1 | 79.1 ± 1.4 | 84.5 ± 1.0 |
| clean (50 Hz) | clean | 90.5 ± 3.2 | 90.3 ± 3.2 | 89.9 ± 3.3 | 88.8 ± 3.6 | 85.1 ± 4.2 |  |
| clean (50 Hz) | aug | 91.8 ± 2.6 | 91.5 ± 2.7 | 90.9 ± 2.7 | 89.1 ± 3.0 | 84.0 ± 3.5 |  |
| webcam: 12 fps + 0.15 + 100 ms | SHARK2+smooth | 70.6 ± 8.4 | 76.8 ± 8.3 | 78.1 ± 8.0 | 77.6 ± 7.5 | 72.8 ± 7.1 | 77.8 ± 7.5 |
| webcam: 12 fps + 0.15 + 100 ms | clean | 75.6 ± 13.0 | 75.6 ± 12.7 | 75.4 ± 12.5 | 74.4 ± 12.2 | 71.3 ± 11.7 |  |
| webcam: 12 fps + 0.15 + 100 ms | aug | 83.4 ± 10.6 | 83.0 ± 10.4 | 82.5 ± 10.2 | 80.6 ± 9.9 | 75.2 ± 9.8 |  |
