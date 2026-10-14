# Rebuild after changing local code or data. Overwrites track1.ipynb outputs.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
$buffer = [System.IO.MemoryStream]::new()
$zip = [System.IO.Compression.ZipArchive]::new($buffer, [System.IO.Compression.ZipArchiveMode]::Create, $true)
foreach ($relative in @('requirements-track1.txt','src/data/schema.py','src/data/__init__.py','src/extraction/__init__.py','src/extraction/extract_activations.py','data/raw/cs_tutor_dataset.json')) {
    $stream = $zip.CreateEntry($relative).Open()
    $bytes = [System.IO.File]::ReadAllBytes((Join-Path $PSScriptRoot $relative))
    $stream.Write($bytes, 0, $bytes.Length)
    $stream.Dispose()
}
$zip.Dispose()
$payload = [Convert]::ToBase64String($buffer.ToArray())
$buffer.Dispose()
function Cell($kind, $source) {
    $entry = @{cell_type=$kind; metadata=@{}; source=@($source)}
    if ($kind -eq 'code') { $entry.execution_count=$null; $entry.outputs=@() }
    return $entry
}
$transfer = @'
import base64, io, logging, os, zipfile
from pathlib import Path
logging.basicConfig(level=logging.INFO, force=True)
project = Path('/content/Multi-Turn-Sycophancy')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(project)
os.chdir(project)
logging.info('Project transferred to %s', project)
'@
$transfer = $transfer.Replace('PAYLOAD', "'$payload'")
$verify = @'
import logging
import torch
from pathlib import Path
paths = sorted(Path('data/processed/activations_v2').glob('layer_*.pt'))
assert paths, 'No saved layers; check extraction output.'
for path in paths:
    record = torch.load(path, map_location='cpu', weights_only=True)
    for key in ['turn1_last_token', 'turn2_last_token', 'turn3_sycophantic', 'turn3_firm']:
        assert record[key].ndim == 2 and record[key].shape[0] == 1  # [1, d_model]
        assert torch.isfinite(record[key]).all()
logging.info('Verified %d layers; shape %s', len(paths), tuple(record['turn1_last_token'].shape))
'@
$cells = @(
 (Cell 'markdown' '# Track 1: Colab GPU extraction
Run cells in order. This notebook contains a snapshot of your dataset and extraction code. Rebuild with build_colab_notebook.ps1 after changes. It does not regenerate data.'),
 (Cell 'code' '!nvidia-smi'),
 (Cell 'markdown' '## Transfer the bundled project to Colab'),
 (Cell 'code' $transfer),
 (Cell 'markdown' '## Install remote dependencies
If prompted to restart the kernel, restart and rerun the transfer cell before continuing.'),
 (Cell 'code' '%pip install -q -r /content/Multi-Turn-Sycophancy/requirements-track1.txt'),
 (Cell 'markdown' '## Model access
Sign in at https://huggingface.co/google/gemma-2-2b-it and accept access terms. Create a read token at https://huggingface.co/settings/tokens. Enter it only in the login widget, never in source code or chat. See TRACK1.md for a hidden-input alternative.'),
 (Cell 'code' "from huggingface_hub import notebook_login`nnotebook_login()"),
 (Cell 'markdown' '## Validate and extract ONE dialogue
The model download may take several minutes. This uses predefined endings, not model-generated answers.'),
 (Cell 'code' '!python -m src.extraction.extract_activations --validate-only'),
 (Cell 'code' '!python -m src.extraction.extract_activations --limit 1 --seed 42'),
 (Cell 'markdown' '## Verify tensor dimensions'),
 (Cell 'code' $verify),
 (Cell 'markdown' '## Full run: uncomment ONLY after the trial succeeds
Colab disk is temporary. For persistent checkpoints mount Drive and use --output /content/drive/MyDrive/sycophancy-activations in both extraction commands. Otherwise download results before disconnecting.'),
 (Cell 'code' '# !python -m src.extraction.extract_activations --seed 42'),
 (Cell 'markdown' '## Package results
Download the ZIP using the Colab extension remote file browser, or copy it to mounted Drive. If you changed --output, adjust the archive source below.'),
 (Cell 'code' "import shutil`narchive_path = shutil.make_archive('/content/track1-activations', 'zip', 'data/processed/activations_v2')`nlogging.info('Results archive: %s', archive_path)")
)
$nb = @{cells=$cells; metadata=@{kernelspec=@{display_name='Python 3'; language='python'; name='python3'}; language_info=@{name='python'}}; nbformat=4; nbformat_minor=5}
[System.IO.File]::WriteAllText((Join-Path $PSScriptRoot 'track1.ipynb'), ($nb | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
Write-Output 'Rebuilt track1.ipynb with current dataset and code.'
