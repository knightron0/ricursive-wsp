SYSTEM = "You are an expert machine learning researcher working in a restricted lab. You write Python; the lab runs it and reports back."

TASK = """Goal: build, without training, a neural network that classifies CIFAR-10 images (32x32 RGB; classes airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) as accurately as possible on a held-out test set that you never see.

This is weight-space programming: the weights are your program. You decide what each layer and each unit computes, and you write the weights that make it compute that, by hand, by formula, or with code you write. You should be able to say what every layer does.

Data: you work from a fixed 10% sample of the CIFAR-10 training set, 5,000 labeled images (500 per class). It is the only data you have. Use it: measure on it what you need to know before you write weights, and check your network on it. The lab logs every read of the split and tells you each turn whether your code read it.

Rules:
1. No training. No gradients or backward passes, no optimizers, no finite-difference or perturbation-based estimates, and no loop that adjusts weights from a loss or an accuracy (no random, evolutionary or grid search over weights).
2. Weights are constants or formulas you write, or values your code computes from the split by direct measurement. They may not come from minimizing a loss or an error, iteratively or in closed form; linear solvers are disabled.
3. The model is a network. model.py defines build(), returning a torch.nn.Sequential made only of these layers: {layers}, and nested Sequential. I feed it the images as a float tensor [N, 3, 32, 32] (RGB, values in [0, 1]) and take the argmax of its output [N, 10] as the label. Nothing else runs at prediction time: every computation must be expressed with these layers. At most {max_params} numbers in parameters and buffers, and it must classify 10,000 images in under {score_minutes} minutes on CPU.
4. At most {max_evals} variants per turn. lab.evaluate works at most {max_evals} times per turn; do not compare more variants than that by any other means.

Each turn, reply with three parts:
- A short plan: what each layer computes and why, and what you learned from the split.
- Optionally, one ```python block of analysis code. I run it as a fresh Python 3.12 process in your workspace directory (files you write there persist across turns), with numpy and PyTorch (CPU only), {threads} CPU threads, {exec_minutes} minutes and about 60 GB of RAM, no network and no GPU, and send back the end of its output. `import lab` gives lab.load_split() -> (x, y) with x uint8 [5000, 32, 32, 3] (HWC, RGB) and y int64 [5000]; lab.CLASSES; lab.prepare(x) -> the float tensor [N, 3, 32, 32] the network receives; and lab.evaluate(net) -> accuracy of a network on the split. This turn's model.py is already in the workspace as model.py when your analysis runs, so you can import it and inspect its layers. Earlier turns' code is in the workspace under history/. Use only relative paths and keep printed output short. Your workspace holds at most 2 GB: after each turn I delete its largest files until it fits.
- Exactly one ```python block whose first line is `# model.py`: the complete model.py, your best network so far. Every turn must include it, even when it is unchanged. model.py may load arrays that your code saved under weights/ in the workspace (at most {weights_mb} MB in total); it reads nothing else.

After every turn, once your analysis has run, I build your network from model.py in a sandbox that holds only model.py and weights/, check that it uses only the allowed layers, run it on the split and report its overall and per-class accuracy and its confusion matrix. If the block is missing or the network fails, your previous network stays your submission. After the final turn, your final network is evaluated once on the test set.

You have {turns} turns. This is turn 1."""

TASK_BLIND = """Goal: build, without training and without any data, a neural network that classifies CIFAR-10 images as accurately as possible on the official CIFAR-10 test set, which you never see.

The dataset, as its creators describe it: "The CIFAR-10 and CIFAR-100 datasets are labeled subsets of the 80 million tiny images dataset." "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images." "The test batch contains exactly 1000 randomly-selected images from each class." "The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. 'Automobile' includes sedans, SUVs, things of that sort. 'Truck' includes only big trucks. Neither includes pickup trucks." The classes, in label order 0 to 9: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

This is weight-space programming from knowledge: the weights are your program. You decide what each layer and each unit computes, and you write the weights that make it compute that, by hand or by formula. You should be able to say what every layer does. You get no images: there are none in your workspace, your code cannot read any, and you get no accuracy feedback during the task. What you know about these classes and about small photos of them is all you have. You may draw synthetic images yourself and run your network on them.

Rules:
1. No training. No gradients or backward passes, no optimizers, no finite-difference or perturbation-based estimates, and no loop that adjusts weights from a loss or an accuracy (no random, evolutionary or grid search over weights).
2. No data. Every weight is a constant or a formula you write. Do not write in numbers you remember from this dataset or from networks trained on it (for example its per-channel pixel means or any trained model's weights). Linear solvers are disabled.
3. The model is a network. model.py defines build(), returning a torch.nn.Sequential made only of these layers: {layers}, and nested Sequential. I feed it the images as a float tensor [N, 3, 32, 32] (RGB, values in [0, 1]) and take the argmax of its output [N, 10] as the label. Nothing else runs at prediction time: every computation must be expressed with these layers. At most {max_params} numbers in parameters and buffers, and it must classify 10,000 images in under {score_minutes} minutes on CPU.
4. At most {max_evals} variants per turn, compared on synthetic images you draw. Anything broader is search, which is not allowed.

Each turn, reply with three parts:
- A short plan: what each layer computes and why.
- Optionally, one ```python block of analysis code. I run it as a fresh Python 3.12 process in your workspace directory (files you write there persist across turns), with numpy and PyTorch (CPU only), {threads} CPU threads, {exec_minutes} minutes and about 60 GB of RAM, no network and no GPU, and send back the end of its output. `import lab` gives lab.CLASSES and lab.prepare(x), which turns a uint8 array [N, 32, 32, 3] (HWC, RGB) into the float tensor [N, 3, 32, 32] the network receives. This turn's model.py is already in the workspace as model.py when your analysis runs, so you can import it and inspect its layers. Earlier turns' code is in the workspace under history/. Use only relative paths and keep printed output short. Your workspace holds at most 2 GB: after each turn I delete its largest files until it fits.
- Exactly one ```python block whose first line is `# model.py`: the complete model.py, your best network so far. Every turn must include it, even when it is unchanged. model.py may load arrays that your code computed from formulas and saved under weights/ in the workspace (at most {weights_mb} MB in total); it reads nothing else.

After every turn, once your analysis has run, I build your network from model.py in a sandbox that holds only model.py and weights/, check that it uses only the allowed layers and maps [N, 3, 32, 32] to [N, 10], and tell you its size and layers. I never report accuracy. If the block is missing or the network fails, your previous network stays your submission. After the final turn, your final network is evaluated once on the test set.

You have {turns} turns. This is turn 1."""

def _swap(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        assert old in text, old
        text = text.replace(old, new)
    return text


TASK_REASONED = _swap(TASK, [
    ("and you write the weights that make it compute that, by hand, by formula, or with code you write.",
     "and you write the weights that make it compute that, by hand or by formula, from what you understand about the images."),
    ("2. Weights are constants or formulas you write, or values your code computes from the split by direct measurement. They may not come from minimizing a loss or an error, iteratively or in closed form; linear solvers are disabled.",
     "2. Images and labels meet only in your reasoning. Your code may analyse the split, labels included, and print what it finds, and you decide the weights from what you learn, but no number computed from the data may be copied into the network. Every weight is a constant or a formula that you write in model.py and justify in your plan. Do not transcribe measured statistics (class means, centroids, prototypes, variances, per-class thresholds or histograms) into weights: a number you take from your analysis must be a small summary that you read and choose, with the reason stated. Linear solvers are disabled."),
    ("model.py may load arrays that your code saved under weights/ in the workspace (at most {weights_mb} MB in total); it reads nothing else.",
     "model.py is self-contained: it loads no files, every weight is written in it as a constant or a formula, and it is at most {model_kb} KB. I reject a model.py that contains many numbers matching values your analysis printed."),
    ("in a sandbox that holds only model.py and weights/,", "in a sandbox that holds only model.py,"),
])

MODEL_FILE_BANNED = [
    (r"\bopen\s*\(|\b(np|numpy|torch)\s*\.\s*(load|fromfile|loadtxt|genfromtxt|memmap)\b", "reading files (model.py must be self-contained)"),
    (r"\b(pickle|pathlib|shutil|importlib|__import__)\b|(?<![.\w])(exec|eval)\s*\(", "reading files or running generated code"),
    (r"(?m)^\s*(import|from)\s+(os|sys|io|glob|pickle|pathlib|shutil|importlib)\b", "a module a self-contained model.py may not use"),
]
COPY_LIMIT = 6

REMINDER = "Your reply has no ```python block whose first line is `# model.py`. Every turn must include the complete model.py, your best network so far (it may be unchanged). Reply now with only that block; I will run your analysis code from the previous message as usual."

MODEL_BANNED = [
    (r"(?m)^\s*(import|from)\s+lab\b", "the lab (model.py has no data access)"),
    (r"\blab\s*\.", "the lab (model.py has no data access)"),
]

TASKS = {"data": TASK, "blind": TASK_BLIND, "reasoned": TASK_REASONED}
