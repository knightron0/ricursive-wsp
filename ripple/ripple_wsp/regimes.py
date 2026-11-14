from dataclasses import dataclass


@dataclass(frozen=True)
class Regime:
    task: str
    focus: str | None
    seed: str
    summary: str


REGIMES = {
    "data": Regime(
        "data", None, "forbidden",
        "Agent sees the 5,000-image split; weights are constants, formulas or direct measurements on the split; no fitting.",
    ),
    "generation": Regime(
        "data", None, "required",
        "Same rules as data, but the agent starts from an earlier network (model.py + weights/) and improves it.",
    ),
    "filters": Regime(
        "data", "filters", "required",
        "A generation whose task adds focus/filters.md: engineer the feature layers for the classes the seed confuses.",
    ),
    "generalize": Regime(
        "data", "filters,generalize", "required",
        "A filters generation that also adds focus/generalize.md: hold out part of the split and judge designs on it.",
    ),
    "blind": Regime(
        "blind", None, "forbidden",
        "No data at all: the official CIFAR-10 description, no images, no accuracy feedback; weights by hand or formula.",
    ),
    "reasoned": Regime(
        "reasoned", None, "forbidden",
        "Agent sees the split, but images and labels meet only in its reasoning: self-contained model.py, no weight files, "
        "copied measurements rejected.",
    ),
}
