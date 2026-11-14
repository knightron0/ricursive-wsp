# First message, regime `blind`

Generated with `python -m ripple_wsp task blind` (defaults: 15 turns, 16 threads). The system message is: "You are an expert machine learning researcher working in a restricted lab. You write Python; the lab runs it and reports back."

````text
Goal: build, without training and without any data, a neural network that classifies CIFAR-10 images as accurately as possible on the official CIFAR-10 test set, which you never see.

The dataset, as its creators describe it: "The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the 80 million tiny images dataset." "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images." "The test batch contains exactly 1000 randomly-selected images from each class." "The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. 'Automobile' includes sedans, SUVs, things of that sort. 'Truck' includes only big trucks. Neither includes pickup trucks." The classes, in label order 0 to 9: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

This is weight-space programming from knowledge: the weights are your program. You decide what each layer and each unit computes, and you write the weights that make it compute that, by hand or by formula. You should be able to say what every layer does. You get no images: there are none in your workspace, your code cannot read any, and you get no accuracy feedback during the task. What you know about these classes and about small photos of them is all you have. You may draw synthetic images yourself and run your network on them.

Rules:
1. No training. No gradients or backward passes, no optimizers, no finite-difference or perturbation-based estimates, and no loop that adjusts weights from a loss or an accuracy (no random, evolutionary or grid search over weights).
2. No data. Every weight is a constant or a formula you write. Do not write in numbers you remember from this dataset or from networks trained on it (for example its per-channel pixel means or any trained model's weights). Linear solvers are disabled.
3. The model is a network. model.py defines build(), returning a torch.nn.Sequential made only of these layers: Conv2d, Linear, ReLU, LeakyReLU, Tanh, Sigmoid, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d, Flatten, BatchNorm1d, BatchNorm2d, Softmax, Identity, and nested Sequential. I feed it the images as a float tensor [N, 3, 32, 32] (RGB, values in [0, 1]) and take the argmax of its output [N, 10] as the label. Nothing else runs at prediction time: every computation must be expressed with these layers. At most 25,000,000 numbers in parameters and buffers, and it must classify 10,000 images in under 5 minutes on CPU.
4. At most 5 variants per turn, compared on synthetic images you draw. Anything broader is search, which is not allowed.

Each turn, reply with three parts:
- A short plan: what each layer computes and why.
- Optionally, one ```python block of analysis code. I run it as a fresh Python 3.12 process in your workspace directory (files you write there persist across turns), with numpy and PyTorch (CPU only), 16 CPU threads, 10 minutes and about 60 GB of RAM, no network and no GPU, and send back the end of its output. `import lab` gives lab.CLASSES and lab.prepare(x), which turns a uint8 array [N, 32, 32, 3] (HWC, RGB) into the float tensor [N, 3, 32, 32] the network receives. This turn's model.py is already in the workspace as model.py when your analysis runs, so you can import it and inspect its layers. Earlier turns' code is in the workspace under history/. Use only relative paths and keep printed output short. Your workspace holds at most 2 GB: after each turn I delete its largest files until it fits.
- Exactly one ```python block whose first line is `# model.py`: the complete model.py, your best network so far. Every turn must include it, even when it is unchanged. model.py may load arrays that your code computed from formulas and saved under weights/ in the workspace (at most 100 MB in total); it reads nothing else.

After every turn, once your analysis has run, I build your network from model.py in a sandbox that holds only model.py and weights/, check that it uses only the allowed layers and maps [N, 3, 32, 32] to [N, 10], and tell you its size and layers. I never report accuracy. If the block is missing or the network fails, your previous network stays your submission. After the final turn, your final network is evaluated once on the test set.

You have 15 turns. This is turn 1.
````
