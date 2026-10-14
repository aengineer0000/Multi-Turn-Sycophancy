# Track 1

Open `track1.ipynb`, select a Colab GPU, and run cells in order. The bundled
project snapshot transfers itself: no separate dataset upload is required.
Install dependencies remotely, accept the Gemma access terms on Hugging Face,
and log in using a read token. If the login widget fails in VS Code, use:

```python
from getpass import getpass
from huggingface_hub import login
login(token=getpass("Hugging Face read token: "), add_to_git_credential=False)
```

The notebook first extracts one dialogue. Verify the resulting tensor shapes
before uncommenting the full-run cell. `data/processed/activations_v2/layer_{L}.pt` contains four CPU float32
`[N, d_model]` tensors matching the midterm interface, plus `topics` and stable
`sample_ids`. Keep paired endings together in Track 2 train/test splits.

`samples/` contains atomic per-item checkpoints. Rerunning resumes them.
`manifest.json` records the dataset hash, model, dependency versions, and
extraction settings; changed settings require a new output directory.
Colab disk is temporary: mount Drive through the extension command and use
`--output /content/drive/MyDrive/sycophancy-activations` in both trial and full
commands to preserve work across sessions. Adjust the verification and ZIP
cells to that directory if you change it.

The official Gemma template requires a user-first conversation. The extractor
adds `Please explain {topic}.` before the dataset's tutor-first context, identically
in both branches. Boundaries are the last content token, excluding turn-closing
tokens. Token prefixes are checked. Shared context states are extracted once
from a context-only pass, avoiding comparisons between different-length fp16
runs. Long sequences
fail rather than truncate. Residuals are extracted without weight centering
or layer-norm folding. Endings are teacher-forced; this does not measure natural
capitulation or prove early prediction.

Core extraction lives under `src/extraction/`, as required by project rules.
The dynamic turn-list helper supports additional turns, but the current
four-tensor midterm export explicitly requires two context turns.

Rebuild the notebook snapshot after local edits (this clears saved outputs):

```powershell
.\build_colab_notebook.ps1
```

CLI from the repository root, after installing `requirements-track1.txt`:

```sh
python -m src.extraction.extract_activations --validate-only
python -m src.extraction.extract_activations --limit 1 --seed 42
python -m src.extraction.extract_activations --seed 42
```
