param(
    [ValidateSet("start", "stop", "status", "logs")]
    [string]$Action = "status"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$dockerExe = "F:\Docker\DockerDesktop\resources\bin\docker.exe"
$dockerConfig = Join-Path $projectRoot "deploy\milvus\docker-client-config"
$composeFile = Join-Path $projectRoot "deploy\mysql\docker-compose.yml"
$environmentFile = Join-Path $projectRoot "deploy\mysql\.env"

if (-not (Test-Path -LiteralPath $dockerExe)) {
    throw "找不到Docker：$dockerExe"
}

switch ($Action) {
    "start" { & $dockerExe --config $dockerConfig compose --env-file $environmentFile -f $composeFile up -d }
    "stop" { & $dockerExe --config $dockerConfig compose --env-file $environmentFile -f $composeFile stop }
    "status" { & $dockerExe --config $dockerConfig compose --env-file $environmentFile -f $composeFile ps }
    "logs" { & $dockerExe --config $dockerConfig compose --env-file $environmentFile -f $composeFile logs --tail 100 }
}

exit $LASTEXITCODE

