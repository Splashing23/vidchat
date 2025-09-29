<#
Creates a virtual environment in backend\venv if it doesn't exist,
installs requirements from requirements.txt (using the venv's python),
creates a .env from env_example.txt if missing, and starts uvicorn.

This script is intended to be invoked from the repository root via the
helper Node script which uses: powershell -NoProfile -ExecutionPolicy Bypass -File start.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Push-Location $scriptDir
try {
    Write-Host "Backend script running in: $scriptDir"

    # Choose Python launcher: prefer 'py -3', then 'python', then 'python3'
    $pythonCmd = $null
    $pythonArgsPrefix = $null

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = 'py'
        $pythonArgsPrefix = '-3'
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = 'python'
        $pythonArgsPrefix = $null
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $pythonCmd = 'python3'
        $pythonArgsPrefix = $null
    } else {
        Write-Error "No Python executable found on PATH. Install Python 3 and ensure 'python' or 'py' is available."
        exit 1
    }

    function Run-Python {
        param(
            [Parameter(Mandatory=$true)] [string[]] $Args
        )
        if ($pythonArgsPrefix) {
            & $pythonCmd $pythonArgsPrefix @Args
        } else {
            & $pythonCmd @Args
        }
    }

    $venvPath = Join-Path $scriptDir 'venv'
    if (-not (Test-Path $venvPath)) {
        Write-Host "Creating virtual environment at $venvPath..."
        Run-Python -Args @('-m','venv',$venvPath)
    } else {
        Write-Host "Virtual environment already exists at $venvPath"
    }

    # Path to the venv python
    $venvPython = Join-Path $venvPath 'Scripts\python.exe'
    if (-not (Test-Path $venvPython)) {
        # fallback - in case of different layout
        $venvPython = Join-Path $venvPath 'bin/python'
    }

    if (-not (Test-Path $venvPython)) {
        Write-Error "Failed to find python in the virtual environment. Expected: $venvPython"
        exit 1
    }

    Write-Host "Installing Python dependencies from requirements.txt using $venvPython..."
    & $venvPython -m pip install --upgrade pip setuptools wheel | Write-Host
    & $venvPython -m pip install -r (Join-Path $scriptDir 'requirements.txt')

    # Create .env from example if missing
    $envFile = Join-Path $scriptDir '.env'
    $envExample = Join-Path $scriptDir 'env_example.txt'
        # Ensure OPENAI_API_KEY is present: prefer environment, then repo .env.local, then backend/.env
        if (-not $env:OPENAI_API_KEY) {
            $repoRoot = Resolve-Path (Join-Path $scriptDir '..')
            $repoEnvLocal = Join-Path $repoRoot '.env.local'
            $backendEnv = Join-Path $scriptDir '.env'

            function Read-KeyFromFile($filePath) {
                if (-not (Test-Path $filePath)) { return $null }
                try {
                    $content = Get-Content $filePath -ErrorAction Stop -Raw
                    $m = [regex]::Match($content, '^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$', 'Multiline')
                    if ($m.Success) { return $m.Groups[1].Value.Trim('"') }
                } catch { return $null }
                return $null
            }

            $found = Read-KeyFromFile $repoEnvLocal
            if (-not $found) { $found = Read-KeyFromFile $backendEnv }
            if ($found) {
                Write-Host "Using OPENAI_API_KEY from file"
                $env:OPENAI_API_KEY = $found
            } else {
                Write-Error "\nERROR: Missing OpenAI API key."
                Write-Error "Set OPENAI_API_KEY in your shell, add it to .env.local in the repository root, or add it to backend/.env."
                exit 1
            }
        }

        # Create .env file if it doesn't exist
        if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
            Write-Host "Creating .env from env_example.txt"
            Copy-Item -Path $envExample -Destination $envFile -Force
            Write-Host "Please edit .env and add your OpenAI API key (OPENAI_API_KEY=...)"
        }

    Write-Host "Starting FastAPI server with uvicorn..."
    # Use the venv python to run uvicorn so the venv packages are used
    & $venvPython -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

} finally {
    Pop-Location
}
