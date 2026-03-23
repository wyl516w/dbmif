$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = $PSScriptRoot
$repoDir = Split-Path -Parent $scriptDir
$pyDir = Join-Path $scriptDir "test"
$pythonExe = "python"

$model = ""
$asrModels = "wenet whisper"
$cuda = ""
$inputDir = Join-Path $repoDir "dataset\gendata"
$outputDir = Join-Path $repoDir "results"
$configPath = ""
$ckptPath = ""
$dryRun = $false

function Show-Usage {
    @"
Usage:
  .\test.ps1 MODEL_NAME [--asr "wenet whisper"|wenet|whisper] [--cuda "0[,1]"]
             [--config path\to\model.yaml] [--ckpt path\to\model.ckpt]
             [--input path\to\input] [--output path\to\output] [--dry-run]
"@
}

function Fail([string]$message, [int]$code = 2) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit $code
}

function Need-File([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        Fail "missing file: $path"
    }
}

function Need-Dir([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Container)) {
        Fail "missing dir: $path"
    }
}

function Test-HasWavs([string]$path) {
    return [bool](Get-ChildItem -Path $path -Recurse -Filter *.wav -File -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function AutoDetect-Config([string]$modelName) {
    $cand1 = Join-Path $repoDir ("config\model_config\{0}.yaml" -f $modelName)
    $cand2 = Join-Path $scriptDir ("yaml\{0}.yaml" -f $modelName)
    if (Test-Path -LiteralPath $cand1) { return $cand1 }
    if (Test-Path -LiteralPath $cand2) { return $cand2 }
    return ""
}

function AutoDetect-Ckpt([string]$modelName) {
    $checkpointsDir = Join-Path $repoDir "checkpoints"
    if (Test-Path -LiteralPath $checkpointsDir) {
        $candidate = Get-ChildItem -Path $checkpointsDir -File -Filter "$modelName*.ckpt" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }

    $logsDir = Join-Path $repoDir "logs"
    if (Test-Path -LiteralPath $logsDir) {
        $candidate = Get-ChildItem -Path $logsDir -Recurse -File -Filter *.ckpt -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*\$modelName-*\version_*\checkpoints\epoch=*-step=*.ckpt" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($candidate) { return $candidate.FullName }
    }

    return ""
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

function Invoke-Step([string]$label, [string[]]$commandParts) {
    Write-Host $label
    if ($dryRun) {
        Write-Host ("DRY-RUN: " + (Format-Command $commandParts))
        return
    }
    & $commandParts[0] @($commandParts[1..($commandParts.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

function Get-Timestamp {
    return Get-Date -Format "yyyy-MM-dd HH:mm:ss"
}

if ($args.Count -lt 1) {
    Show-Usage
    exit 1
}

if ($args[0] -eq "-h" -or $args[0] -eq "--help") {
    Show-Usage
    exit 0
}

$model = $args[0]
for ($i = 1; $i -lt $args.Count; ) {
    switch ($args[$i]) {
        "--asr" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --asr" 1 }
            $asrModels = $args[$i + 1]
            $i += 2
        }
        "--cuda" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --cuda" 1 }
            $cuda = $args[$i + 1]
            $i += 2
        }
        "--config" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --config" 1 }
            $configPath = $args[$i + 1]
            $i += 2
        }
        "--ckpt" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --ckpt" 1 }
            $ckptPath = $args[$i + 1]
            $i += 2
        }
        "--input" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --input" 1 }
            $inputDir = $args[$i + 1]
            $i += 2
        }
        "--output" {
            if ($i + 1 -ge $args.Count) { Fail "missing value for --output" 1 }
            $outputDir = $args[$i + 1]
            $i += 2
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

$enhDir = Join-Path $outputDir $model
$cleanDir = Join-Path $outputDir "clean"
$refTxt = Join-Path $enhDir "refer.txt"
$metCsv = Join-Path $enhDir "metrics.csv"

$skipInfer = $false
if ($model -eq "ac" -or $model -eq "bc") {
    $skipInfer = $true
}

if ($cuda) {
    $env:CUDA_VISIBLE_DEVICES = $cuda
}

if (-not (Test-Path -LiteralPath $outputDir)) {
    if ($dryRun) {
        Write-Host "[dry-run] Would create output dir: $outputDir"
    } else {
        New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
        Write-Host "[mkdir] Created output dir: $outputDir"
    }
}

if (-not $configPath) {
    $configPath = AutoDetect-Config $model
}
if (-not $ckptPath) {
    $ckptPath = AutoDetect-Ckpt $model
}

if (-not $dryRun) {
    if ($skipInfer) {
        Write-Host ">>> MODEL='$model' -> skipping inference (using existing $enhDir)"
        Need-Dir $enhDir
        if (-not (Test-HasWavs $enhDir)) {
            Fail "No .wav files found under '$enhDir' to evaluate."
        }
    } else {
        Need-Dir $inputDir
        if (-not $configPath) { Fail "cannot auto-detect config for MODEL=$model" }
        if (-not $ckptPath) { Fail "cannot auto-detect ckpt for MODEL=$model" }
        Need-File $configPath
        Need-File $ckptPath
    }
}

$asrArr = @($asrModels -split "\s+" | Where-Object { $_ })

Write-Host "============================================================"
Write-Host ("test.ps1 - " + (Get-Timestamp))
Write-Host "Model:    $model   (skip_infer=$([int]$skipInfer))"
Write-Host ("ASR:      " + ($asrArr -join " "))
if ($cuda) {
    Write-Host "CUDA:     $cuda"
} else {
    Write-Host "CUDA:     <unchanged>"
}
if (-not $skipInfer) {
    Write-Host "Config:   $configPath"
    Write-Host "Ckpt:     $ckptPath"
}
Write-Host "Input:    $inputDir"
Write-Host "Output:   $outputDir"
Write-Host "Enhanced: $enhDir"
Write-Host "Clean:    $cleanDir"
Write-Host "PyDir:    $pyDir"
if ($dryRun) {
    Write-Host "Mode:     dry-run"
}
Write-Host "============================================================"

Push-Location $repoDir
try {
    if (-not $skipInfer) {
        Invoke-Step "[1/5] infer_and_save.py" @(
            $pythonExe,
            (Join-Path $pyDir "infer_and_save.py"),
            "--config", $configPath,
            "--ckpt", $ckptPath,
            "--input", $inputDir,
            "--output", $outputDir,
            "-n", $model
        )
    }

    Invoke-Step "[2/5] generate_refer.py" @(
        $pythonExe,
        (Join-Path $pyDir "generate_refer.py"),
        "-e", (Join-Path $enhDir ""),
        "-s", $refTxt
    )

    Invoke-Step "[3/5] compute_metrics_csv.py" @(
        $pythonExe,
        (Join-Path $pyDir "compute_metrics_csv.py"),
        "-c", $cleanDir,
        "-e", $enhDir,
        "-o", $metCsv,
        "-m", "classic", "dns", "origin"
    )

    $asrCommand = @(
        $pythonExe,
        (Join-Path $pyDir "batch_asr.py"),
        "-i", (Join-Path $enhDir ""),
        "-r", $refTxt,
        "-m"
    )
    $asrCommand += $asrArr
    $asrCommand += @(
        "-o", (Join-Path $enhDir "")
    )
    Invoke-Step ("[4/5] batch_asr.py  (models: " + ($asrArr -join " ") + ")") $asrCommand

    Write-Host "[5/5] cer_to_csv.py (per ASR engine)"
    foreach ($engine in $asrArr) {
        $asrTxt = Join-Path $enhDir ("{0}.txt" -f $engine)
        $cerCsv = Join-Path $enhDir ("{0}_cer.csv" -f $engine)
        if ($dryRun) {
            Write-Host ("DRY-RUN: " + (Format-Command @(
                $pythonExe,
                (Join-Path $pyDir "cer_to_csv.py"),
                "-r", $refTxt,
                "-p", $asrTxt,
                "-o", $cerCsv
            )))
            continue
        }
        if (-not (Test-Path -LiteralPath $asrTxt -PathType Leaf)) {
            Write-Host "[warn] Missing ASR hypothesis for '$engine': $asrTxt (skipping CER)"
            continue
        }
        & $pythonExe (Join-Path $pyDir "cer_to_csv.py") -r $refTxt -p $asrTxt -o $cerCsv
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        Write-Host "  wrote: $cerCsv"
    }
} finally {
    Pop-Location
}

Write-Host "------------------------------------------------------------"
Write-Host "Done. Outputs:"
Write-Host "  Refer:   $refTxt"
Write-Host "  Metrics: $metCsv"
foreach ($engine in $asrArr) {
    Write-Host ("  ASR:     " + (Join-Path $enhDir ("{0}.txt" -f $engine)))
    Write-Host ("  CER CSV: " + (Join-Path $enhDir ("{0}_cer.csv" -f $engine)))
}
Write-Host "============================================================"
