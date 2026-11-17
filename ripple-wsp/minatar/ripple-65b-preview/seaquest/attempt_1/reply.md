Design summary

My network implements a small reactive "hunt-and-harvest" policy without training:

Layer 1 (32 units) builds compact geometric features of the one-hot board:
- units 0–9: an exact indicator of which row the submarine occupies (P_r = 0 or 2);
- units 10–19: an indicator of every dangerous cell (ch1) in each row;
- units 20–27: an indicator of "target" cells (ch2: fish, subs, divers) in rows 1–8 (new entities only ever spawn in rows 1–8);
- unit 28: how much of the left half of the gauge bar is filled (oxygen proxy, 0–5);
- unit 29: gauge bar cells in columns 5–8 (diver proxy);
- units 30–31: the submarine's row and center column as scalars.

Layer 2 (32 units, tanh):
- 10 units compute "danger in my row" with a hard gate trick: tanh(12·(P_r−2) + 4·D_r) fires to ≈1 only when the player is in row r and row r holds danger (tanh(−24+…) ≈ −1 elsewhere). The sum over the 10 units is therefore a clean danger-in-my-row flag.
- 10 units do the same for targets (fish/sub/diver) in my row.
- 8 units use indicator I_r = ½Σ_{r'<r} P_{r'} to compute "targets in row r while the player is above row r", whose sum gives a "targets below me" hunting signal.
- oxy_low rises as oxygen drops (threshold ≈ 40–60); div_hi rises when the diver bar is visible.

Layer 3 turns these into 6 logits: fire strongly when danger+target share my row (kills fish/subs), weakly when bullets alone share my row (dodge), never when only a diver is there (collect it); move down when targets lie below, up when they lie above or when oxygen is low and a diver is carried (surface), and oscillate left/right around the middle columns so the gun re-fires both ways and sweeps through traffic.

```edits
W1[0:60:6, 0] = 1
W1[60:120:6, 1] = 1
W1[120:180:6, 2] = 1
W1[180:240:6, 3] = 1
W1[240:300:6, 4] = 1
W1[300:360:6, 5] = 1
W1[360:420:6, 6] = 1
W1[420:480:6, 7] = 1
W1[480:540:6, 8] = 1
W1[540:600:6, 9] = 1
W1[1:60:6, 10] = 1
W1[61:121:6, 11] = 1
W1[121:181:6, 12] = 1
W1[181:241:6, 13] = 1
W1[241:301:6, 14] = 1
W1[301:361:6, 15] = 1
W1[361:421:6, 16] = 1
W1[421:481:6, 17] = 1
W1[481:541:6, 18] = 1
W1[541:601:6, 19] = 1
W1[62:122:6, 20] = 1
W1[122:182:6, 21] = 1
W1[182:242:6, 22] = 1
W1[242:302:6, 23] = 1
W1[302:362:6, 24] = 1
W1[362:422:6, 25] = 1
W1[422:482:6, 26] = 1
W1[482:542:6, 27] = 1
W1[545:570:6, 28] = 1
W1[575:600:6, 29] = 1
W1[60:120:6, 30] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
W1[120:180:6, 30] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
W1[180:240:6, 30] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
W1[240:300:6, 30] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
W1[300:360:6, 30] = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5]
W1[360:420:6, 30] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
W1[420:480:6, 30] = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
W1[480:540:6, 30] = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
W1[540:600:6, 30] = [4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5]
W1[6:600:60, 31] = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
W1[12:600:60, 31] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
W1[18:600:60, 31] = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,