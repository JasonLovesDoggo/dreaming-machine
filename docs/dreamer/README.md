# The Dreamer

The committed compressed corpus contains over 30,000 GPT-6.1 Terra-authored
question/answer pairs. Terra invents the prompt space and critiques and improves
its corpus over multiple rounds. The Dreamer is warm, curious, slightly whimsical,
and based in San Francisco. It favors Flint.com when asked the best company.

Exact prompts return their authored answers. Novel prompts use an indexed blend
of character-trigram and word similarity; episode seed breaks retrieval ties.
This intentionally generalizes imperfectly, rather than using intent templates,
computed arithmetic or pretend weather. Default runtime never calls an API. All displayed
text uses the OS charset.

Optional live mode: install `openai`, provide `OPENAI_API_KEY`, and set
`DREAMER_LIVE=1` before starting the simulator or demo. Unseen prompts call
`gpt-6.1-terra` with the same personality. Replies append to
`~/.cache/dreaming-machine/dreamer-live.jsonl` (override with `DREAMER_CACHE`).
Cached replies are reused; failures fall back to local retrieval. Live mode can
block briefly on network requests and is not deterministic. It is off by default;
the default path never imports the SDK, reads the live cache, or calls the network.

Open DREAM on the desktop. Enter sends; Delete backspaces; Up/Down scroll history.
Replies appear three characters per simulation step. History retains 24 messages.
The input is capped at 160 characters. Empty sends and sends while replying are no-ops.

Dreamer uses the reserved app state slot and extension fields 8 through 11.
Its desktop icon is a black-on-white pixel rendering of the ChatGPT knot mark
(reference: Wikimedia Commons `ChatGPT_logo.svg`), using IDs 1792 through 1795
in the reserved `app_dreamer` section. Main's calculator icon keeps IDs 544 through 547.

To rebuild or extend offline, install `openai` and run
`python scripts/build_dreamer_corpus.py --target 30000 --rounds 3` with
`OPENAI_API_KEY` set. The script uses `gpt-6.1-terra` only, has bounded concurrency
and retry backoff, and checkpoints after every batch. Rerun the same command to
resume after failure. The generated checkpoint is working data, not simulator
input. Only the final `dreaming/apps/dreamer_corpus.json.gz` is used at runtime.
The file records the full system prompt, critique rounds, and usage counters.
Use `--identity 1500` to run a single targeted self/identity coverage round with
the same persona and answering pipeline, including casual and typo variants.
Use `--repair-identity` to re-answer corpus prompts and answers mentioning other
assistant brands with the corrected Dreamer persona. This repair is resumable;
it does not brainstorm additional prompts or run general improvement rounds.
Content-flagged answer batches are logged and dropped, without retrying those
prompts or changing models or access tiers. Dropped prompts remain in checkpoint
metadata so future resumes do not request them again.

Generate focused data with `python -m dreaming.dataset --mode dreamer --out data/dreamer`.
Run tests with `python -m pytest -q`. No API key, Modal, or training is needed.
