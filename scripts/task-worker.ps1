param(
    [ValidateSet("start", "stop", "status", "logs")]
    [string]$Action = "status"
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "E:\anaconda\envs\mamba\python.exe"
$runtimeDir = Join-Path $projectRoot "storage\task_workers"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project Python not found: $pythonExe"
}
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$workers = @(
    @{ Name = "gpu"; Queue = "gpu" },
    @{ Name = "maintenance"; Queue = "maintenance" }
)

function Get-WorkerPidPath($name) { Join-Path $runtimeDir "$name.pid" }

function Get-LiveWorker($name) {
    $pidPath = Get-WorkerPidPath $name
    if (-not (Test-Path -LiteralPath $pidPath)) { return $null }
    $workerPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$workerPid" -ErrorAction SilentlyContinue
    if ($process -and $process.CommandLine -like "*task_queue.celery_app*") { return $process }
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    return $null
}

switch ($Action) {
    "start" {
        foreach ($worker in $workers) {
            if (Get-LiveWorker $worker.Name) {
                Write-Host "$($worker.Name) worker is already running"
                continue
            }
            $stdout = Join-Path $runtimeDir "$($worker.Name).stdout.log"
            $stderr = Join-Path $runtimeDir "$($worker.Name).stderr.log"
            $arguments = @(
                "-m", "celery", "-A", "task_queue.celery_app:celery_app", "worker",
                "--loglevel=INFO", "--pool=solo", "--queues=$($worker.Queue)",
                "--hostname=$($worker.Name)@%h"
            )
            $process = Start-Process -FilePath $pythonExe -ArgumentList $arguments `
                -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
                -RedirectStandardOutput $stdout -RedirectStandardError $stderr
            Set-Content -LiteralPath (Get-WorkerPidPath $worker.Name) -Value $process.Id
            Write-Host "$($worker.Name) worker started, PID=$($process.Id)"
        }
    }
    "stop" {
        foreach ($worker in $workers) {
            $process = Get-LiveWorker $worker.Name
            if ($process) { Stop-Process -Id $process.ProcessId -Force }
            Remove-Item -LiteralPath (Get-WorkerPidPath $worker.Name) -Force -ErrorAction SilentlyContinue
            Write-Host "$($worker.Name) worker stopped"
        }
    }
    "status" {
        foreach ($worker in $workers) {
            $process = Get-LiveWorker $worker.Name
            [pscustomobject]@{
                Worker = $worker.Name
                Queue = $worker.Queue
                Running = [bool]$process
                PID = if ($process) { $process.ProcessId } else { $null }
            }
        }
    }
    "logs" {
        foreach ($worker in $workers) {
            Write-Host "=== $($worker.Name) ==="
            $stderr = Join-Path $runtimeDir "$($worker.Name).stderr.log"
            if (Test-Path -LiteralPath $stderr) { Get-Content -LiteralPath $stderr -Tail 80 }
        }
    }
}
