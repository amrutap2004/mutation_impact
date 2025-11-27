## Windows Setup Guide (No Pre-installed Tools)

Follow these steps on a fresh Windows 10/11 laptop to run this project and the web app command `mi-web` (invoked via `.\.venv\Scripts\mi-web`). All commands below are PowerShell unless noted.

### 1) Install Python 3.11 (recommended)
- Go to the official Python downloads page: `https://www.python.org/downloads/windows/`
- Download the latest Python 3.11.x Windows installer (64-bit).
- Run the installer:
  - Check "Add Python to PATH"
  - Choose "Customize installation" → keep defaults → "Install for all users" if possible

Verify installation:
```powershell
python --version
```
You should see `Python 3.11.x`. If `python` is not found, close and reopen PowerShell.

### 2) Install Microsoft C++ Build Tools (needed for some Python packages)
Some scientific dependencies (e.g., `numpy`, `scipy`, `weasyprint` transitive deps) require build tools.

- Install the Build Tools:
  - Go to `https://visualstudio.microsoft.com/visual-cpp-build-tools/`
  - Download and run the installer.
  - In the installer, select "Desktop development with C++" workload and complete installation.

If installation asks for reboot, reboot before continuing.

### 3) Install GTK/WeasyPrint runtime dependencies (for PDF rendering)
We use WeasyPrint for PDF generation which needs GTK/Cairo/Pango/Harfbuzz on Windows.

- Easiest way: install GStreamer/GTK runtime via MSYS/WinGet or official binaries.
  - Option A (recommended): WinGet installs
    ```powershell
    winget install GnuPG.Glib
    winget install Gnome.GTK
    ```
    If either package is unavailable, use Option B.
  - Option B: Install GTK3 runtime bundle from `https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer`.
    - Download the latest installer (x64) and run it (default path, add to PATH when asked).

After installation, close and reopen PowerShell to refresh PATH.

Note: If you do not need PDF generation, you can skip this step; HTML reports will still work. The web app will run without WeasyPrint’s PDF runtime.

### 4) Clone or copy the project
If you received a ZIP, extract to a path without spaces if possible (e.g., `C:\projects\mutation_impact`). If using Git:
```powershell
git clone https://github.com/your-org/mutation_impact.git
cd mutation_impact
```

### 5) Create and activate a virtual environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
 .\.venv\Scripts\Activate.ps1
```

Confirm the venv is active: your prompt should start with `(.venv)`.

### 6) Install project dependencies
The project is packaged with `pyproject.toml`. Install in editable mode:
```powershell
pip install --upgrade pip
pip install -e .
```

This installs dependencies including: biopython, numpy, pandas, scikit-learn, scipy, gemmi, matplotlib, jinja2, requests, freesasa, Flask, WeasyPrint, xgboost, seaborn, joblib.

If you see build errors for wheels (e.g., `scipy`), ensure C++ Build Tools are installed (step 2). If WeasyPrint complains on PDF features, ensure GTK runtime is installed (step 3) or continue without PDF support.

### 7) Verify CLI entry points
The installation provides these commands in the venv’s `Scripts` directory:
- `mi-validate` → `mutation_impact.cli:main`
- `mi-run` → `mutation_impact.pipeline_cli:main`
- `mi-web` → `mutation_impact.web.app:main`

List to confirm they exist:
```powershell
dir .\.venv\Scripts\mi-*.exe
```
On Windows, you may also find `.cmd` or no extension launchers. You can always run the module directly:
```powershell
python -m mutation_impact.web.app
```

### 8) Quick validation
```powershell
mi-validate --seq MATK --mut A2T
```
If the command is not found, run it explicitly from the venv Scripts path:
```powershell
 .\.venv\Scripts\mi-validate --seq MATK --mut A2T
```

### 9) Run the web app
```powershell
mi-web
```
If the PATH launcher isn’t recognized, use the explicit path:
```powershell
 .\.venv\Scripts\mi-web
```
or use Python module execution:
```powershell
python -m mutation_impact.web.app
```

The app will print a local URL (e.g., `http://127.0.0.1:5000`). Open it in your browser. If the port is busy, set `FLASK_RUN_PORT`:
```powershell
$env:FLASK_RUN_PORT=5050
mi-web
```

### 10) Run the pipeline from CLI (optional)
```powershell
mi-run --seq MKTFFVAI... --mut A123T --uniprot-id P05067 --out report.html
start report.html
```

### 11) Troubleshooting
- If `gemmi` fails to install: ensure you are on Python 3.9–3.11 and pip is up to date. Try `pip install --only-binary=:all: gemmi`.
- If `xgboost` fails on Windows: ensure Visual C++ runtime is present. Try `pip install xgboost==1.7.6`.
- If `WeasyPrint` complains about missing DLLs: ensure GTK runtime is installed and on PATH, or skip PDF features.
- If activation script is blocked: make sure you ran `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force` in PowerShell.

### 12) Uninstall / clean
```powershell
deactivate
Remove-Item -Recurse -Force .\.venv
```

You can reinstall by repeating steps 5–6.


