The policy network is hand-designed to intercept the ball at the bottom row. It uses the first hidden layer to detect:
- the ball’s column (units 0–9),
- the paddle’s column (units 10–19),
- whether the ball is moving down-right or down-left in rows 3–8 (units 20–31).

The second hidden layer computes the difference between the paddle’s current column and the predicted bottom-row column of the ball, assuming no future horizontal reflections. The third layer maps this difference to left/right/no-op actions: it moves left if the paddle is to the right of the predicted target, right if to the left, and stays otherwise.

```edits
# ---------- W1: B_c units (ball column) ----------
W1[3:600:60, 0] = 1
W1[9:600:60, 1] = 1
W1[15:600:60, 2] = 1
W1[21:600:60, 3] = 1
W1[27:600:60, 4] = 1
W1[33:600:60, 5] = 1
W1[39:600:60, 6] = 1
W1[45:600:60, 7] = 1
W1[51:600:60, 8] = 1
W1[57:600:60, 9] = 1

# ---------- W1: P_c units (paddle column) ----------
W1[540, 10] = 1
W1[546, 11] = 1
W1[552, 12] = 1
W1[558, 13] = 1
W1[564, 14] = 1
W1[570, 15] = 1
W1[576, 16] = 1
W1[582, 17] = 1
W1[588, 18] = 1
W1[594, 19] = 1

# ---------- W1: M_DR_y units (moving down-right) ----------
W1[189:243:6, 20] = 1
W1[124:178:6, 20] = 1
W1[249:303:6, 21] = 1
W1[184:238:6, 21] = 1
W1[309:363:6, 22] = 1
W1[244:298:6, 22] = 1
W1[369:423:6, 23] = 1
W1[304:358:6, 23] = 1
W1[429:483:6, 24] = 1
W1[364:418:6, 24] = 1
W1[489:543:6, 25] = 1
W1[424:478:6, 25] = 1

# ---------- W1: M_DL_y units (moving down-left) ----------
W1[183:237:6, 26] = 1
W1[130:184:6, 26] = 1
W1[243:297:6, 27] = 1
W1[190:244:6, 27] = 1
W1[303:357:6, 28] = 1
W1[250:304:6, 28] = 1
W1[363:417:6, 29] = 1
W1[310:364:6, 29] = 1
W1[423:477:6, 30] = 1
W1[370:424:6, 30] = 1
W1[483:537:6, 31] = 1
W1[430:484:6, 31] = 1

# ---------- b1 ----------
b1[20:32] = -1

# ---------- W2: first hidden unit computes d = paddle_x - target_x ----------
W2[0, 0:32] = [
    0, -1.313, -2.626, -3.939, -5.252, -6.565, -7.878, -9.191, -10.504, -11.817,
    0, 1.313, 2.626, 3.939, 5.252, 6.565, 7.878, 9.191, 10.504, 11.817,
    -6.565, -5.252, -3.939, -2.626, -1.313, 0,
    6.565, 5.252, 3.939, 2.626, 1.313, 0
]

# ---------- W3 and b3: map d to actions ----------
W3[0, 1] = 10       # left
W3[0, 3] = -10      # right
b3[1] = -1
b3[3] = -1
b3[2] = -100
b3[4] = -100
b3[5] = -100
```