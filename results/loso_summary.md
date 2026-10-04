# D1 leave-one-subject-out summary (9 held-out subjects: 0, 1, 2, 4, 5, 6, 7, 8, 9)

Top-1 word accuracy (%), mean ± sd over held-out subjects; every decoder on traces resampled to 50 Hz.
`aug > clean` = held-out subjects where the augmented model beats the clean one.

| Condition | SHARK2+smooth@50Hz | clean@50Hz | aug@50Hz | aug − clean | aug > clean |
|---|---|---|---|---|---|
| clean (50 Hz) | 67.8 ± 20.3 | 86.3 ± 12.9 | 86.6 ± 15.9 | +0.3 | 4/9 |
| 30 fps | 70.3 ± 21.3 | 85.8 ± 12.6 | 86.5 ± 15.8 | +0.7 | 6/9 |
| 12 fps | 71.6 ± 21.6 | 82.8 ± 13.4 | 85.4 ± 15.4 | +2.7 | 8/9 |
| jitter 0.1 | 66.6 ± 20.0 | 86.5 ± 13.0 | 86.9 ± 15.7 | +0.3 | 4/9 |
| jitter 0.2 | 63.1 ± 18.5 | 86.3 ± 12.8 | 86.8 ± 15.7 | +0.5 | 4/9 |
| jitter 0.3 | 58.1 ± 16.3 | 85.5 ± 13.0 | 86.4 ± 15.0 | +0.9 | 6/9 |
| webcam: 12 fps + 0.15 + 100 ms | 69.0 ± 9.1 | 74.6 ± 12.5 | 82.9 ± 10.0 | +8.3 | 9/9 |
| harsh: 10 fps + 0.3 + 100 ms | 61.3 ± 7.8 | 69.3 ± 13.4 | 78.5 ± 11.3 | +9.2 | 9/9 |

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
| 9 | 1081 | 56.7 | 66.9 | 79.4 |

Word-frequency prior (score = log-lik + λ·log p(word)), top-1 (%), mean over 9 subjects.
SHARK2's λ is tuned per held-out subject on the other subjects (chosen: 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0).

| Condition | Decoder | λ=0 | λ=0.25 | λ=0.5 | λ=1 | λ=2 | SHARK2 tuned λ |
|---|---|---|---|---|---|---|---|
| clean (50 Hz) | SHARK2+smooth | 67.9 ± 20.7 | 75.5 ± 22.1 | 77.5 ± 21.9 | 77.5 ± 21.2 | 73.0 ± 18.2 | 77.5 ± 21.2 |
| clean (50 Hz) | clean | 86.3 ± 12.9 | 86.2 ± 12.7 | 86.1 ± 11.8 | 85.2 ± 11.3 | 81.9 ± 10.5 |  |
| clean (50 Hz) | aug | 86.6 ± 15.9 | 86.6 ± 14.9 | 86.3 ± 14.1 | 84.9 ± 13.0 | 80.6 ± 10.8 |  |
| webcam: 12 fps + 0.15 + 100 ms | SHARK2+smooth | 69.0 ± 9.2 | 75.4 ± 8.9 | 77.0 ± 8.1 | 76.9 ± 7.3 | 72.4 ± 6.8 | 77.1 ± 7.4 |
| webcam: 12 fps + 0.15 + 100 ms | clean | 74.6 ± 12.5 | 74.6 ± 12.2 | 74.5 ± 12.0 | 73.7 ± 11.6 | 70.6 ± 11.1 |  |
| webcam: 12 fps + 0.15 + 100 ms | aug | 82.9 ± 10.0 | 82.7 ± 9.7 | 82.2 ± 9.5 | 80.5 ± 9.3 | 75.3 ± 9.2 |  |
