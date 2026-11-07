# dreaming-computer

A deterministic tile-based micro OS (desktop, file explorer + trash, Paint, Solitaire, Doom-lite) driven by a
scripted "smart user", used to train a world model that predicts the next screen from the current screen + input.

- Screen: 32x24 tiles of 8x8 px (256x192). Every tile comes from a fixed atlas (`dreaming/tiles.py`), so a frame is
  exactly 768 integer tokens. Text uses a 3x5 font with two characters per tile.
- Doom-lite: 160x120 textured raycaster, encoded as 160 column tokens (wall type, side, height, texture u) + 4 sprite
  tokens + a meta token (gun flash, viewport position).
- Cursor: separate x/y tokens. Action per step: (mouse x, mouse y, left button, key).
- Model (`dreaming/model.py`): transformer over tiles + cursor + action + doom tokens + recurrent memory tokens +
  OS-state query tokens (auxiliary loss). Predicts all next-frame tokens in one pass.

```bash
python -m dreaming.dataset --out data/local --steps 400000          # ~40s on 8 cores, ~16 MB
python -m dreaming.train --data data/local --out runs/local         # GPU recommended
python -m dreaming.demo --ckpt runs/local/ckpt.pt --port 8000       # real vs neural, side by side

# Modal
modal run modal_app.py::gen_data --total-steps 20000000
modal run modal_app.py::train --gpus 8 --iters 30000
modal run modal_app.py::fetch
```
