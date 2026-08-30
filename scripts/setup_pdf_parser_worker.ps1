param(
    [string]$BootstrapPython = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$workerPython = Join-Path $projectRoot ".venv-pdf-parser\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $workerPython)) {
    if (-not $BootstrapPython) {
        $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCommand) {
            $BootstrapPython = $pythonCommand.Source
        }
    }
    if (-not $BootstrapPython -or -not (Test-Path -LiteralPath $BootstrapPython -PathType Leaf)) {
        throw "未找到可用 Python。请使用 -BootstrapPython 指定 Python 3.10+ 的 python.exe 路径。"
    }
    & $BootstrapPython -m venv (Join-Path $projectRoot ".venv-pdf-parser")
}

& $workerPython -m pip install -r (Join-Path $projectRoot "requirements_pdf_parser_worker.txt")
& $workerPython -c "import pymupdf, pdfplumber; from rapidocr import RapidOCR; print('Lightweight PDF parser ready')"
