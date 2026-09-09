# Запуск интерактивной проверки из PowerShell.
# Стрелки: выбрать видео в ТЕКУЩЕЙ папке и формат сжатия.
# После анализа отчёт печатается в этом же окне.
#
#   cd C:\путь\к\папке\с\видео
#   C:\Users\user\Desktop\practoring_test_tech\start.ps1
#
# Или из папки проекта (там тоже можно положить видео):
#   .\start.ps1

$ErrorActionPreference = "Stop"

try {
    chcp 65001 | Out-Null
} catch {}

$OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::InputEncoding = $OutputEncoding
[Console]::OutputEncoding = $OutputEncoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:PYTHONPATH = $Root

# Ищем видео в текущей папке, откуда вызвали скрипт.
& $Python -m app --interactive --dir (Get-Location).Path @args
exit $LASTEXITCODE
