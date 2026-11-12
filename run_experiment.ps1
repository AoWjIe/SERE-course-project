$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path $PSScriptRoot).Path
$python = (Get-Command python -ErrorAction Stop).Source
$script = Join-Path $projectRoot 'course_reproduction\run_experiment.py'
& $python $script --root $projectRoot
exit $LASTEXITCODE
