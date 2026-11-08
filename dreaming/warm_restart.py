"""Resume training with an explicitly restarted learning-rate schedule."""
import argparse
import math
import sys

from dreaming.picture_bank import install_picture_bank


def restart_lr(it, args, restart_iter, warmup, min_lr_frac):
    if restart_iter >= args.iters or warmup < 1 or not 0 <= min_lr_frac <= 1:
        raise ValueError('require restart_iter < iters, warmup >= 1, 0 <= min_lr_frac <= 1')
    if it < restart_iter:
        raise ValueError('iteration precedes restart: checkpoint was not resumed')
    progress = min(1, (it - restart_iter) / (args.iters - restart_iter))
    return args.lr * min(1, (it - restart_iter + 1) / warmup) * (
        min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    from dreaming import train
    parser = argparse.ArgumentParser()
    parser.add_argument('--restart-iter', type=int, required=True)
    parser.add_argument('--restart-warmup', type=int, required=True)
    parser.add_argument('--min-lr-frac', type=float, required=True)
    parser.add_argument('--picture-data', default='')
    parser.add_argument('--picture-frac', type=float, default=0.)
    parser.add_argument('--picture-holdout-episodes', type=int, default=0)
    own, rest = parser.parse_known_args()
    if rest[:1] != ['--']:
        parser.error('separate training arguments with --')
    sys.argv = ['dreaming.train', *rest[1:]]
    args = train.parse_args()
    restart_lr(own.restart_iter, args, own.restart_iter, own.restart_warmup, own.min_lr_frac)
    train.lr_at = lambda it, a: restart_lr(it, a, own.restart_iter, own.restart_warmup, own.min_lr_frac)
    install_picture_bank(train, own.picture_data, own.picture_frac, own.picture_holdout_episodes)
    train.main()


if __name__ == '__main__':
    main()
