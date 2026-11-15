The policy is a vertical-only survival strategy. The player never moves horizontally; treasures cross the entire row, so simply standing in a safe row will collect any treasure that passes through the player's column. The network detects the player's row, enemy rows, and whether the rows above/below are safe. If an enemy is in the current row, the player moves vertically to the nearest adjacent row without an enemy, avoiding the top/bottom edges unless forced. If the current row is safe, the player stays, collecting treasures automatically. Edge rows are treated carefully: from row 0 the player only moves down if row 1 is safe, and from row 9 only moves up if row 8 is safe.

```edits
# W1: player row indicators (indices 0..9)
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

# W1: enemy row indicators (indices 10..17 correspond to rows 1..8)
W1[61:121:6, 10] = 1
W1[121:181:6, 11] = 1
W1[181:241:6, 12] = 1
W1[241:301:6, 13] = 1
W1[301:361:6, 14] = 1
W1[361:421:6, 15] = 1
W1[421:481:6, 16] = 1
W1[481:541:6, 17] = 1

# b1 remains all zeros

# W2: hidden layer 2 weights for F_up_y (y=1..8)
W2[0, 1] = 10
W2[0, 10] = 10
b2[0] = -15
W2[1, 2] = 10
W2[1, 11] = 10
W2[1, 10] = -10
b2[1] = -15
W2[2, 3] = 10
W2[2, 12] = 10
W2[2, 11] = -10
b2[2] = -15
W2[3, 4] = 10
W2[3, 13] = 10
W2[3, 12] = -10
b2[3] = -15
W2[4, 5] = 10
W2[4, 14] = 10
W2[4, 13] = -10
b2[4] = -15
W2[5, 6] = 10
W2[5, 15] = 10
W2[5, 14] = -10
b2[5] = -15
W2[6, 7] = 10
W2[6, 16] = 10
W2[6, 15] = -10
b2[6] = -15
W2[7, 8] = 10
W2[7, 17] = 10
W2[7, 16] = -10
b2[7] = -15

# W2: hidden layer 2 weights for F_down_y (y=1..8)
W2[8, 1] = 10
W2[8, 10] = 10
W2[8, 11] = -10
b2[8] = -15
W2[9, 2] = 10
W2[9, 11] = 10
W2[9, 12] = -10
b2[9] = -15
W2[10, 3] = 10
W2[10, 12] = 10
W2[10, 13] = -10
b2[10] = -15
W2[11, 4] = 10
W2[11, 13] = 10
W2[11, 14] = -10
b2[11] = -15
W2[12, 5] = 10
W2[12, 14] = 10
W2[12, 15] = -10
b2[12] = -15
W2[13, 6] = 10
W2[13, 15] = 10
W2[13, 16] = -10
b2[13] = -15
W2[14, 7] = 10
W2[14, 16] = 10
W2[14, 17] = -10
b2[14] = -15
# F_down_8: down from row 8 only if row 7 is blocked (enemy in row 7)
W2[15, 8] = 10
W2[15, 17] = 10
W2[15, 16] = 10
b2[15] = -25

# W2: edge safety features
# h2[16]: player in row 0 and no enemy in row 1
W2[16, 0] = 10
W2[16, 10] = -10
b2[16] = -5
# h2[17]: player in row 9 and no enemy in row 8
W2[17, 9] = 10
W2[17, 17] = -10
b2[17] = -5

# W3: up logit (action 2)
W3[0:8, 2] = 10
W3[17, 2] = 100
b3[2] = 5

# W3: down logit (action 4)
W3[8:16, 4] = 10
W3[16, 4] = 100
b3[4] = 0

# W3: biases for other actions
b3[0] = -70
b3[1] = -100
b3[2] = 5
b3[3] = -100
b3[4] = 0
b3[5] = -100
```