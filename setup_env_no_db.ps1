param(
    [string]$VenvName = ".venv"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Resolve and move to project root (where this script is located).
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not available in PATH. Install Python 3.11+ and try again."
}

$requirementsFile = Join-Path $projectRoot "requirements.txt"
if (-not (Test-Path $requirementsFile)) {
    throw "requirements.txt not found at $requirementsFile"
}

Write-Host "Creating virtual environment: $VenvName"
python -m venv $VenvName

$pythonExe = Join-Path $projectRoot "$VenvName\Scripts\python.exe"
$pipExe = Join-Path $projectRoot "$VenvName\Scripts\pip.exe"

if (-not (Test-Path $pythonExe) -or -not (Test-Path $pipExe)) {
    throw "Virtual environment creation failed. Expected executables not found in $VenvName\\Scripts."
}

Write-Host "Upgrading pip tooling..."
& $pythonExe -m pip install --upgrade pip setuptools wheel

# Skip database-related packages while keeping the rest of requirements.
$excludedPackages = @(
    "sqlalchemy",
    "alembic",
    "aiosqlite",
    "asyncpg",
    "motor",
    "pymongo"
)

$filteredRequirements = New-Object System.Collections.Generic.List[string]
foreach ($line in Get-Content $requirementsFile) {
    $trimmed = $line.Trim()

    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
        continue
    }

    # Extract package name from forms like: package==x, package>=x, package[extra]==x
    $match = [regex]::Match($trimmed, '^[A-Za-z0-9_.-]+')
    if (-not $match.Success) {
        continue
    }

    $packageName = $match.Value.ToLowerInvariant()
    if ($excludedPackages -contains $packageName) {
        continue
    }

    $filteredRequirements.Add($trimmed)
}

if ($filteredRequirements.Count -eq 0) {
    throw "No installable dependencies found after filtering database packages."
}

$tempRequirements = Join-Path $env:TEMP ("requirements_no_db_{0}.txt" -f [guid]::NewGuid().ToString())
$filteredRequirements | Set-Content -Path $tempRequirements -Encoding UTF8

try {
    Write-Host "Installing non-database dependencies..."
    & $pipExe install -r $tempRequirements
}
finally {
    if (Test-Path $tempRequirements) {
        Remove-Item $tempRequirements -Force
    }
}

Write-Host ""
Write-Host "Done. Virtual environment is ready at: $VenvName"
Write-Host "Activate with: .\\$VenvName\\Scripts\\Activate.ps1"