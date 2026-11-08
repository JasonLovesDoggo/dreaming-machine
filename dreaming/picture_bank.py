"""Opt-in picture-bank sampling without changing the trainer."""
from pathlib import Path

import numpy as np


def batch_counts(batch, extra_frac, picture_frac):
    if not (0 <= extra_frac and 0 < picture_frac and extra_frac + picture_frac < 1):
        raise ValueError('require positive picture fraction and extra_frac + picture_frac < 1')
    cats, pictures = round(batch * extra_frac), round(batch * picture_frac)
    primary = batch - cats - pictures
    if pictures < 1 or primary < 1:
        raise ValueError('batch must contain at least one primary and picture sample')
    return primary, cats, pictures


def picture_paths(directories):
    paths = set()
    for directory in directories.split(','):
        path = Path(directory).resolve()
        if not directory or 'heldout' in directory.lower() or 'heldout' in str(path).lower() or 'train' not in path.parts:
            raise ValueError('picture paths must resolve under /train/ and never contain heldout')
        if not path.is_dir():
            raise ValueError(f'picture directory does not exist: {directory}')
        for shard in path.glob('*.npz'):
            resolved = shard.resolve()
            if 'train' not in resolved.parent.parts or 'heldout' in str(shard).lower() or 'heldout' in str(resolved).lower():
                raise ValueError('picture shard resolves outside /train/ or contains heldout')
            paths.add(str(resolved))
    if not paths:
        raise ValueError('no picture train shards')
    return sorted(paths)


class PictureWindows:
    def __init__(self, mixed, pictures, picture_frac):
        self.mixed, self.pictures, self.picture_frac = mixed, pictures, picture_frac

    def batch(self, batch):
        primary, cats, pictures = batch_counts(batch, self.mixed.frac, self.picture_frac)
        banks = [self.mixed.primary.batch(primary)]
        if cats:
            banks.append(self.mixed.extra.batch(cats))
        banks.append(self.pictures.batch(pictures))
        return {key: np.concatenate([bank[key] for bank in banks]) for key in banks[0]}


def install_picture_bank(train, directories, fraction=0., holdout_episodes=0):
    if fraction == 0:
        return
    if not 0 < fraction < 1:
        raise ValueError('picture fraction must be in (0,1)')
    if type(holdout_episodes) is not int or holdout_episodes < 0:
        raise ValueError('picture holdout episodes must be a nonnegative int')
    paths = picture_paths(directories)
    original = train.load_data

    def load_data(args, rank, world):
        if not args.extra_data:
            raise ValueError('pictures require --extra-data and a MixedWindows sampler')
        batch_counts(args.batch, args.extra_frac, fraction)
        mixed = original(args, rank, world)
        if not isinstance(mixed, train.MixedWindows):
            raise ValueError('pictures require the original MixedWindows sampler')
        data = train.load_training_shards(paths, holdout_episodes=holdout_episodes)
        pictures = train.Windows(data, args.unroll, np.random.default_rng(9012 + rank), args.sampler_v2, None)
        return PictureWindows(mixed, pictures, fraction)

    train.load_data = load_data
