# QASense local demo

## Prerequisites

- Windows and Microsoft Edge
- Python 3.14
- NVIDIA CUDA GPU for the verified VisionQA path
- Protected model artifacts restored at the paths in `reports/QASENSE_MIGRATION_MANIFEST.md`
- Git LFS for the CodeLocator v2 model when cloning a published repository

Create the integration environment:

```powershell
py -3.14 -m venv .venv-integration
.\.venv-integration\Scripts\python.exe -m pip install -r requirements-product.txt
.\.venv-integration\Scripts\python.exe -m playwright install msedge
```

Create the three model environments using their pinned requirement files. Install VisionQA Torch from the CUDA 12.6 PyTorch index as documented in `requirements-visionqa.txt`.

## Run the complete fixture analysis

```powershell
.\.venv-integration\Scripts\python.exe scripts\run_qasense_demo.py
```

The command analyzes `tests/fixtures/runtime_fixture.html`, invokes all three protected production model adapters, saves browser and product artifacts, and prints the result path. No credentials or network service are required; the fixture intentionally emits one page error, one console error, and one failed request.

## Start API and dashboard

```powershell
.\.venv-integration\Scripts\python.exe scripts\serve_qasense.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`. The dashboard provides Web Analysis, Code Analysis, and Full Analysis. Saved runs show original and annotated screenshots, risk-ranked files, exact CodeLocator lines, findings, recommendations, test suggestions, regression priorities, and expandable technical evidence.

Start a run through the API:

```powershell
$fixture = [System.Uri]::new((Resolve-Path tests\fixtures\runtime_fixture.html).Path).AbsoluteUri
$body = @{ analysis_type = 'web'; url = $fixture } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/api/runs -ContentType application/json -Body $body
```

Run Code Analysis on the local fixture:

```powershell
$body = @{
  analysis_type = 'code'
  source_path = (Resolve-Path tests\fixtures\CheckoutService.java).Path
  max_source_files = 1
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/api/runs -ContentType application/json -Body $body
```

For Full Analysis, use `analysis_type = 'full'` and include both `url` and `source_path`. The API reports real stages while the run is active; it does not generate synthetic progress percentages.

## Tests

```powershell
.\.venv-integration\Scripts\python.exe -m pytest -q tests\product tests\adapters tests\contracts tests\integration tests\visionqa\test_matching.py tests\visionqa\test_inference_contract.py tests\visual_defects\test_baseline.py
```

Model training scripts are not part of this demo workflow.
