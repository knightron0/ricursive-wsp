"""Train the CIFAR-10 ResNet-18 source model for the cifar10-to-cifar100 task.

This is ordinary gradient training, done once, outside the agent. The
scratch-cifar10 task never sees this checkpoint.
    python train_base.py            # -> checkpoints/resnet18_cifar10.pt
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from wsp import data
from wsp.evaluation import evaluate, summarize
from wsp.model import ResNet18, default_device


def augment(x):
    """Random horizontal flip and random translation of up to 4 pixels (zero padded)."""
    n = len(x)
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    theta = torch.zeros(n, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, :, 2] = torch.randint(-4, 5, (n, 2), device=x.device).float() * 2 / 32
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode='nearest', padding_mode='zeros', align_corners=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('checkpoints/resnet18_cifar10.pt'))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    images, labels = data.load_split('cifar10', 'train', args.data_dir)
    val = data.load_split('cifar10', 'val', args.data_dir)
    classes = data.class_names('cifar10', args.data_dir)
    model = ResNet18(10).to(args.device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    steps = args.epochs * ((len(labels) + args.batch_size - 1) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=0.25)

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        order = torch.randperm(len(labels))
        for start in range(0, len(labels), args.batch_size):
            index = order[start:start + args.batch_size]
            x = augment(data.normalize(images[index].to(args.device))).contiguous(memory_format=torch.channels_last)
            loss = F.cross_entropy(model(x), labels[index].to(args.device), label_smoothing=0.1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        metrics = evaluate(weights, val, classes, args.device)
        print(f'epoch {epoch}: loss {loss.item():.3f}, val accuracy {metrics["accuracy"]:.4f}, '
              f'{time.time() - started:.0f}s', flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(weights, args.out)
    args.out.with_suffix('.json').write_text(json.dumps(
        {'args': {k: str(v) for k, v in vars(args).items()}, 'val': summarize(metrics)}, indent=2) + '\n')
    print(f'Saved {args.out} (val accuracy {metrics["accuracy"]:.4f})')


if __name__ == '__main__':
    main()
