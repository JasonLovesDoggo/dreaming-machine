"""Run the browser demo locally.
  python -m dreaming.demo [--ckpt runs/local/ckpt_005000.pt] [--data data/local] [--port 8000]
Without --ckpt only the real computer runs."""
import argparse
import os
import uvicorn
from .web import create_app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='', help='checkpoint file; its directory is offered in the checkpoint picker')
    ap.add_argument('--data', default=None, help='shard directory for the training-data viewer')
    ap.add_argument('--port', type=int, default=8000)
    a = ap.parse_args()
    ckpt_dir = name = None
    device = 'cpu'
    if a.ckpt:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt_dir, name = os.path.dirname(os.path.abspath(a.ckpt)), os.path.basename(a.ckpt)
    uvicorn.run(create_app(a.data, ckpt_dir, device, name), host='0.0.0.0', port=a.port)


if __name__ == '__main__':
    main()
