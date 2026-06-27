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

# Projekt dziala na Pythonie 3.11-3.13. 3.14+ NIE jest wspierany: numpy 2.2.6 nie ma
# jeszcze wheeli dla 3.14, wiec pip probowalby kompilowac ze zrodel i pada na Windows.
# Dlatego NIE uzywamy "py -3" (to wybiera NAJNOWSZY Python, czyli moze 3.14) tylko
# szukamy pierwszej wspieranej wersji: preferujemy najnowsza dzialajaca (3.13).
$supported = @("3.13", "3.12", "3.11")
$pyVer = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in $supported) {
        & py "-$v" --version *> $null
        if ($LASTEXITCODE -eq 0) { $pyVer = $v; break }
    }
}
if (-not $pyVer) {
    throw "Nie znaleziono wspieranego Pythona (3.11/3.12/3.13). Masz zapewne tylko 3.14+, dla ktorego brakuje wheeli (numpy 2.2.6). Zainstaluj Pythona 3.13: https://www.python.org/downloads/"
}

Write-Host "==> Tworze srodowisko wirtualne .venv na Pythonie $pyVer ..." -ForegroundColor Cyan
& py "-$pyVer" -m venv .venv

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
