param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [switch]$SkipExisting
)

$ErrorActionPreference = 'Stop'
$knowledgeRoot = Join-Path $ProjectRoot 'data\knowledge_docs'
$reportRoot = Join-Path $ProjectRoot 'data\knowledge_sources'

$sources = @(
    @{
        Category='01_纯红外设备'
        File='GuideIR_IPT640M纯红外测温机芯.pdf'
        Url='https://www.guideir.cn/Cn/Skippower/downloadFile?id=460&mid=45'
    },
    @{
        Category='02_纯可见光设备'
        File='HIKROBOT_MV-CA050-12UC纯可见光工业相机.pdf'
        Url='https://www.hikrobotics.com/cn2/source/vision/document/2023/6/9/MV-CA050-12UMUC_20230510.pdf'
    },
    @{
        Category='03_红外可见光双模态设备'
        File='GuideIR_MC-FT红外可见光双模态枪机.pdf'
        Url='https://www.guideir.cn/Cn/Skippower/downloadFile?id=530&mid=45'
    }
)

$results = @()

foreach ($source in $sources) {
    $outputDir = Join-Path $knowledgeRoot $source.Category
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $target = Join-Path $outputDir $source.File
    $status = 'downloaded'
    $note = ''

    if ($SkipExisting -and (Test-Path -LiteralPath $target)) {
        $status = 'existing'
        $note = 'download skipped; existing file checked'
    }
    else {
        & curl.exe -L --fail --retry 2 --connect-timeout 20 -A 'Mozilla/5.0' -o $target $source.Url
        if ($LASTEXITCODE -ne 0) {
            $status = 'failed'
            $note = "curl exit code $LASTEXITCODE"
            if (Test-Path -LiteralPath $target) {
                Remove-Item -LiteralPath $target -Force
            }
        }
    }

    if (Test-Path -LiteralPath $target) {
        $stream = [System.IO.File]::OpenRead($target)
        try {
            $header = New-Object byte[] 5
            [void]$stream.Read($header, 0, 5)
            $signature = [System.Text.Encoding]::ASCII.GetString($header)
        }
        finally {
            $stream.Dispose()
        }

        if ($signature -ne '%PDF-') {
            $status = 'invalid'
            $note = 'response is not a PDF'
            Remove-Item -LiteralPath $target -Force
        }
    }

    $size = if (Test-Path -LiteralPath $target) { (Get-Item -LiteralPath $target).Length } else { 0 }
    $results += [pscustomobject]@{
        category = $source.Category
        filename = $source.File
        status = $status
        bytes = $size
        source_url = $source.Url
        note = $note
    }
}

New-Item -ItemType Directory -Force -Path $reportRoot | Out-Null
$reportPath = Join-Path $reportRoot 'mambadfuse_device_download_report.csv'
$results | Export-Csv -LiteralPath $reportPath -NoTypeInformation -Encoding UTF8
$results | Format-Table category, filename, status, bytes -AutoSize
