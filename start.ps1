$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Write-Host '请先按 README.md 安装依赖并创建 .venv。'
    exit 1
}
& $projectPython -m streamlit run app.py
