# setup.ps1 - jednorazowa instalacja srodowiska deweloperskiego (Windows / PowerShell).
# Tworzy lokalny .venv i instaluje WSZYSTKIE zaleznosci (runtime + dev), w tym mlflow.
#
# Uzycie:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Po instalacji aktywuj srodowisko:  .\.venv\Scripts\Activate.ps1
# i wszystko (app, trening, make mlflow, testy, pylint) dziala z jednego zrodla.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Wybierz dostepny launcher Pythona (py -3 jest pewniejszy na Windows niz samo "python").
$pythonCmd = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else { "python" }

Write-Host "==> Tworze srodowisko wirtualne .venv ..." -ForegroundColor Cyan
Invoke-Expression "$pythonCmd -m venv .venv"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Nie znaleziono $venvPython - utworzenie .venv nie powiodlo sie."
}

Write-Host "==> Aktualizuje pip ..." -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

Write-Host "==> Instaluje zaleznosci z requirements-dev.txt (runtime + dev + mlflow) ..." -ForegroundColor Cyan
& $venvPython -m pip install -r requirements-dev.txt

Write-Host ""
Write-Host "Gotowe." -ForegroundColor Green
Write-Host "1) Aktywuj srodowisko:   .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "2) W PyCharm wskaz interpreter:  .\.venv\Scripts\python.exe" -ForegroundColor Green
Write-Host "3) Przyklady:  make train  |  make mlflow  |  make test  |  make lint" -ForegroundColor Green
