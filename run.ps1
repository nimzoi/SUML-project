# run.ps1 - uruchamia aplikacje (Streamlit UI) z lokalnego .venv. Windows / PowerShell.
# Wymaga wczesniejszej instalacji:  .\setup.ps1
#
# Uzycie:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# UI laduje model lokalnie, wiec API nie jest wymagane do dzialania wyceny.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Brak .venv - najpierw zainstaluj zaleznosci:" -ForegroundColor Red
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "Uruchamiam aplikacje (Streamlit UI) -> http://localhost:8501" -ForegroundColor Cyan
Write-Host "(Ctrl+C konczy dzialanie.)" -ForegroundColor DarkGray
& $venvPython -m streamlit run app/ui.py
