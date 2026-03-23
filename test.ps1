$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "scripts\test.ps1"
& $scriptPath @args
exit $LASTEXITCODE
