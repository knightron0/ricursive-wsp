from ripple_wsp.net import prepare

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_split():
    raise RuntimeError("this task has no data: there are no images to load")


def evaluate(*args, **kwargs):
    raise RuntimeError("this task has no data: there is nothing to evaluate on")
