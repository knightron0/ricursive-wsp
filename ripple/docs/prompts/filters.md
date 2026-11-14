# First message, regime `filters`

Generated with `python -m ripple_wsp task filters` (defaults: 15 turns, 16 threads). The system message is: "You are an expert machine learning researcher working in a restricted lab. You write Python; the lab runs it and reports back."

````text
Goal: build, without training, a neural network that classifies CIFAR-10 images (32x32 RGB; classes airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) as accurately as possible on a held-out test set that you never see.

This is weight-space programming: the weights are your program. You decide what each layer and each unit computes, and you write the weights that make it compute that, by hand, by formula, or with code you write. You should be able to say what every layer does.

Data: you work from a fixed 10% sample of the CIFAR-10 training set, 5,000 labeled images (500 per class). It is the only data you have. Use it: measure on it what you need to know before you write weights, and check your network on it. The lab logs every read of the split and tells you each turn whether your code read it.

Rules:
1. No training. No gradients or backward passes, no optimizers, no finite-difference or perturbation-based estimates, and no loop that adjusts weights from a loss or an accuracy (no random, evolutionary or grid search over weights).
2. Weights are constants or formulas you write, or values your code computes from the split by direct measurement. They may not come from minimizing a loss or an error, iteratively or in closed form; linear solvers are disabled.
3. The model is a network. model.py defines build(), returning a torch.nn.Sequential made only of these layers: Conv2d, Linear, ReLU, LeakyReLU, Tanh, Sigmoid, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, BatchNorm1d, BatchNorm2d, Softmax, Identity, and nested Sequential. I feed it the images as a float tensor [N, 3, 32, 32] (RGB, values in [0, 1]) and take the argmax of its output [N, 10] as the label. Nothing else runs at prediction time: every computation must be expressed with these layers. At most 25,000,000 numbers in parameters and buffers, and it must classify 10,000 images in under 5 minutes on CPU.
4. At most 5 variants per turn. lab.evaluate works at most 5 times per turn; do not compare more variants than that by any other means.

Each turn, reply with three parts:
- A short plan: what each layer computes and why, and what you learned from the split.
- Optionally, one ```python block of analysis code. I run it as a fresh Python 3.12 process in your workspace directory (files you write there persist across turns), with numpy and PyTorch (CPU only), 16 CPU threads, 10 minutes and about 60 GB of RAM, no network and no GPU, and send back the end of its output. `import lab` gives lab.load_split() -> (x, y) with x uint8 [5000, 32, 32, 3] (HWC, RGB) and y int64 [5000]; lab.CLASSES; lab.prepare(x) -> the float tensor [N, 3, 32, 32] the network receives; and lab.evaluate(net) -> accuracy of a network on the split. This turn's model.py is already in the workspace as model.py when your analysis runs, so you can import it and inspect its layers. Earlier turns' code is in the workspace under history/. Use only relative paths and keep printed output short. Your workspace holds at most 2 GB: after each turn I delete its largest files until it fits.
- Exactly one ```python block whose first line is `# model.py`: the complete model.py, your best network so far. Every turn must include it, even when it is unchanged. model.py may load arrays that your code saved under weights/ in the workspace (at most 100 MB in total); it reads nothing else.

After every turn, once your analysis has run, I build your network from model.py in a sandbox that holds only model.py and weights/, check that it uses only the allowed layers, run it on the split and report its overall and per-class accuracy and its confusion matrix. If the block is missing or the network fails, your previous network stays your submission. After the final turn, your final network is evaluated once on the test set.

You do not start from scratch. An earlier agent working under these same rules built the network below; it is your submission now (turn 0), and its weights files are already in your workspace under weights/. Build on it and improve it.
```python
<the seed's model.py>
```
<the seed's split score: size, layers, accuracy, per-class accuracy, confusion matrix>

This generation's focus is the feature layers. In the network above, the convolution is a generic filter bank that is the same for every task, and the classes are told apart only by the measured readout: nothing in the feature layers was designed for these ten classes. Engineer them. Use the split to find what separates the classes the network confuses most (its confusion matrix shows which), decide what visual cue would tell each pair apart, and write filters and layers that compute those cues. You may add layers, combine responses from earlier layers, and remove or replace filters that contribute little. For every filter or layer you add or change, state in your plan which cue it computes and which classes it is meant to separate, and check on the split whether it does.

You have 15 turns. This is turn 1.
````
