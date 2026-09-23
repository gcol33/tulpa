# Detached launcher for route_coverage.R (gcol33/tulpa#862, #865).
# One process per true field scale so the three levels fill at comparable depth
# and an interruption leaves balanced coverage rather than one complete level.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File run_coverage.ps1 0.7 1:30
param(
  [Parameter(Mandatory = $true)][string]$Sigma,
  [string]$Seeds = "1:30"
)

$Repo   = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$OutDir = Join-Path $PSScriptRoot "route_coverage"
$RunDir = Join-Path $OutDir "logs"
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$RHome = (Get-ChildItem 'C:\Program Files\R' -Directory |
          Sort-Object Name -Descending | Select-Object -First 1).FullName
$Rscript = Join-Path $RHome 'bin\Rscript.exe'

$tag = "s$Sigma"
Set-Location $Repo
& $Rscript (Join-Path $PSScriptRoot 'route_coverage.R') $OutDir $Seeds $Sigma `
    *> (Join-Path $RunDir "$tag.log")
