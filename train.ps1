$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoDir = $PSScriptRoot
$pythonExe = "python"

$cuda = "0"
$modelCfgPath = Join-Path $repoDir "config\model_config"
$modelCfg = "DBMIF.yaml"
$logDir = Join-Path $repoDir "logs\paper"
$name = "DBMIF"
$epoch = 100
$batchSize = 16
$recommendedGpuSpace = 200000
$timeForWaitingGpuSpace = 60
$skipGpuCheck = $false
$dryRun = $false

function Show-Usage {
    @"
Usage:
  .\train.ps1 [--cuda 0] [--model-cfg-path .\config\model_config] [--model-cfg DBMIF.yaml]
             [--log-dir .\logs\paper] [--name DBMIF] [--epoch 100] [--batch-size 16]
             [--recommended-gpu-space 200000] [--time-for-waiting-gpu-space 60]
             [--skip-gpu-check] [--dry-run]
"@
}

function Fail([string]$message, [int]$code = 2) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit $code
}

function Format-Command([string[]]$commandParts) {
    return ($commandParts | ForEach-Object {
        if ($_ -match '[\s"]') {
            '"' + ($_ -replace '"', '\"') + '"'
        } else {
            $_
        }
    }) -join " "
}

function Get-FreeGpuMemory([string]$visibleDevices) {
    $gpuId = ($visibleDevices -split ",")[0].Trim()
    $result = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id=$gpuId
    if ($LASTEXITCODE -ne 0 -or -not $result) {
        Fail "failed to query GPU memory with nvidia-smi"
    }
    return [int]($result | Select-Object -First 1).Trim()
}

for ($i = 0; $i -lt $args.Count; ) {
    switch ($args[$i]) {
        "--cuda" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --cuda" 1 }
            $cuda = $args[$i + 1]
            $i += 2
        }
        "--model-cfg-path" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --model-cfg-path" 1 }
            $modelCfgPath = $args[$i + 1]
            $i += 2
        }
        "--model-cfg" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --model-cfg" 1 }
            $modelCfg = $args[$i + 1]
            $i += 2
        }
        "--log-dir" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --log-dir" 1 }
            $logDir = $args[$i + 1]
            $i += 2
        }
        "--name" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --name" 1 }
            $name = $args[$i + 1]
            $i += 2
        }
        "--epoch" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --epoch" 1 }
            $epoch = [int]$args[$i + 1]
            $i += 2
        }
        "--batch-size" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --batch-size" 1 }
            $batchSize = [int]$args[$i + 1]
            $i += 2
        }
        "--recommended-gpu-space" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --recommended-gpu-space" 1 }
            $recommendedGpuSpace = [int]$args[$i + 1]
            $i += 2
        }
        "--time-for-waiting-gpu-space" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --time-for-waiting-gpu-space" 1 }
            $timeForWaitingGpuSpace = [int]$args[$i + 1]
            $i += 2
        }
        "--skip-gpu-check" {
            $skipGpuCheck = $true
            $i += 1
        }
        "--dry-run" {
            $dryRun = $true
            $i += 1
        }
        "-h" {
            Show-Usage
            exit 0
        }
        "--help" {
            Show-Usage
            exit 0
        }
        default {
            Fail "unknown arg: $($args[$i])" 1
        }
    }
}

$env:CUDA_VISIBLE_DEVICES = $cuda
$resolvedModelCfg = Join-Path $modelCfgPath $modelCfg

Write-Host "CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
Write-Host "model_cfg_path: $modelCfgPath"
Write-Host "model_cfg: $modelCfg"
Write-Host "log_dir: $logDir"
Write-Host "name: $name"
Write-Host "epoch: $epoch"
Write-Host "batch_size: $batchSize"

if (-not $skipGpuCheck) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        Fail "nvidia-smi not found; use --skip-gpu-check to bypass the GPU memory check"
    }

    $gpuSpace = Get-FreeGpuMemory $env:CUDA_VISIBLE_DEVICES
    Write-Host "GPU space available: $gpuSpace MB"

    while ($gpuSpace -lt $recommendedGpuSpace) {
        Write-Host "Waiting for GPU space to be larger than $recommendedGpuSpace MB"
        for ($waiting = $timeForWaitingGpuSpace; $waiting -gt 0; $waiting--) {
            Write-Host "$waiting seconds left"
            Start-Sleep -Seconds 1
        }
        $gpuSpace = Get-FreeGpuMemory $env:CUDA_VISIBLE_DEVICES
        Write-Host "GPU space available: $gpuSpace MB"
    }
} else {
    Write-Host "GPU check skipped."
}

$command = @(
    $pythonExe,
    "main.py",
    "-model_cfg", $resolvedModelCfg,
    "-e", "$epoch",
    "-l", $logDir,
    "-n", $name,
    "-b", "$batchSize"
)

Write-Host "training $modelCfg with $epoch epochs and logging to $logDir\$name"
Write-Host (Format-Command $command)

if ($dryRun) {
    Write-Host "Dry run only; training command not executed."
    exit 0
}

Push-Location $repoDir
try {
    & $command[0] @($command[1..($command.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} finally {
    Pop-Location
}
