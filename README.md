# ARGUS

## Local Python Setup

On Windows, use the project-local virtual environment so ARGUS dependencies stay isolated from your global Python install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/base.txt
python scripts/health_check.py
```

You can also run the health check without activating first:

```powershell
.\.venv\Scripts\python scripts\health_check.py
```
