param(
    [string]$Python312Path = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $PSScriptRoot ".venv"
$wheel = Get-ChildItem (Join-Path $root "dist\\pythusa-*.whl") |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $wheel) {
    throw "No built wheel found under dist/. Run 'python -m build' first."
}

if ($Python312Path) {
    & $Python312Path -m venv $venv
}
else {
    py -3.12 -m venv $venv
}

$python = Join-Path $venv "Scripts\\python.exe"

& $python -m pip install --upgrade pip
& $python -m pip install $wheel.FullName

Push-Location $PSScriptRoot
try {
    & $python ".\\smoke_test.py"
}
finally {
    Pop-Location
}
