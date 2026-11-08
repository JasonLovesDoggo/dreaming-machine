# The Dreamer

The local corpus contains over 6,000 simple arithmetic and conversational prompts.
Weather covers 28 cities, including Cincinnati and San Francisco, and is fixed by
episode seed and city. Replies never use a network service. Unknown questions and
unsafe requests get explicit fallbacks. All displayed text uses the OS charset.

Open DREAM on the desktop. Enter sends; Delete backspaces; Up/Down scroll history.
Replies appear three characters per simulation step. History retains 24 messages.
The input is capped at 160 characters. Empty sends and sends while replying are no-ops.

The calendar starts on Thursday, January 1, 2026 at noon; each OS step advances one
second. The existing taskbar clock remains its original static 12:00 label.
Dreamer uses the reserved app state slot and extension fields 8 through 11.

Generate focused data with `python -m dreaming.dataset --mode dreamer --out data/dreamer`.
Run tests with `python -m pytest -q`. No API key, Modal, or training is needed.
