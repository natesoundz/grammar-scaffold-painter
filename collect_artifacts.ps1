# collect_artifacts.ps1
# Batch artifact collection for transformation compilation.
#
# Usage:
#   .\collect_artifacts.ps1
#   .\collect_artifacts.ps1 -InputFile my_inputs.txt -MaxCount 200
#
# Output: artifacts.jsonl (appended each run)

param(
    [string]$InputFile  = "inputs.txt",
    [string]$Artifacts  = "artifacts.jsonl",
    [string]$Substrate  = "substrate.min.bin",
    [int]   $MaxCount   = 500,
    [int]   $Layers     = 5,
    [int]   $Select     = 6
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $InputFile)) {
    Write-Error "Input file not found: $InputFile"
    exit 1
}

# Clear the output file so we start fresh each run
"" | Set-Content $Artifacts -Encoding UTF8
Write-Host "Cleared $Artifacts"
Write-Host "Collecting from $InputFile (max $MaxCount sentences)..."
Write-Host ""

$count   = 0
$skipped = 0
$start   = Get-Date

Get-Content $InputFile | ForEach-Object {
    $line = $_.Trim()

    # Skip blank lines and comment lines
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }

    if ($count -ge $MaxCount) {
        return
    }

    $count++

    python listen_instrumented.py $line `
        --substrate $Substrate `
        --artifacts $Artifacts `
        --layers    $Layers    `
        --select    $Select    `
        --quiet

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  FAILED on: $line"
        $skipped++
    }

    if ($count % 50 -eq 0) {
        $elapsed = (Get-Date) - $start
        $rate    = $count / $elapsed.TotalSeconds
        $eta     = ($MaxCount - $count) / [Math]::Max($rate, 0.01)
        Write-Host ("  [{0,3}] elapsed {1:F0}s  rate {2:F1}/s  eta {3:F0}s" -f `
            $count, $elapsed.TotalSeconds, $rate, $eta)
    }
}

$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host ("Done: {0} artifacts collected, {1} skipped, {2:F1}s elapsed" -f `
    $count, $skipped, $elapsed.TotalSeconds)
Write-Host "Output: $Artifacts"
