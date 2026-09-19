[CmdletBinding()]
# Downloads the pinned, precompiled dependency wheels (see requirements.txt) for
# a given platform/Python version and unpacks them into the add-on's local deps
# directory, without touching Blender's bundled Python installation. Automates
# "Option 2" from README.md's Installing Dependencies section, for CI and for
# developers not running the exact Python version Blender bundles.
param(
    [string] $DepsPath = '',
    [string] $Platform = 'win_amd64',
    [string] $PythonVersion = '313',
    [string] $Implementation = 'cp',
    [string] $Abi = 'cp313'
)

$ErrorActionPreference = 'Stop'

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
$RepositoryRoot = (Resolve-Path (Join-Path $ScriptRoot '..')).Path
$RequirementsFile = Join-Path $RepositoryRoot 'requirements.txt'

if ([string]::IsNullOrWhiteSpace($DepsPath)) {
    $DepsPath = Join-Path $RepositoryRoot 'deps'
}
$ResolvedDeps = [System.IO.Path]::GetFullPath($DepsPath)

if (-not (Test-Path -LiteralPath $RequirementsFile -PathType Leaf)) {
    throw "Could not find requirements file: $RequirementsFile"
}

$WheelDir = Join-Path ([System.IO.Path]::GetTempPath()) ("rwt-wheels-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $WheelDir -Force | Out-Null

try {
    python -m pip download `
        --platform $Platform `
        --python-version $PythonVersion `
        --implementation $Implementation `
        --abi $Abi `
        --only-binary=:all: `
        --no-deps `
        -d $WheelDir `
        -r $RequirementsFile
    if ($LASTEXITCODE -ne 0) {
        throw "pip download failed with exit code $LASTEXITCODE."
    }

    if (Test-Path -LiteralPath $ResolvedDeps) {
        Remove-Item -LiteralPath $ResolvedDeps -Recurse -Force
    }
    New-Item -ItemType Directory -Path $ResolvedDeps -Force | Out-Null

    $Wheels = Get-ChildItem -LiteralPath $WheelDir -Filter '*.whl' -File
    if (@($Wheels).Count -eq 0) {
        throw "No wheels were downloaded into $WheelDir"
    }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach ($Wheel in $Wheels) {
        # Wheels are ZIP files; Expand-Archive requires a .zip extension, so
        # extract directly instead of renaming.
        [System.IO.Compression.ZipFile]::ExtractToDirectory($Wheel.FullName, $ResolvedDeps)
    }
}
finally {
    if (Test-Path -LiteralPath $WheelDir) {
        Remove-Item -LiteralPath $WheelDir -Recurse -Force
    }
}

Write-Output "Installed dependencies into $ResolvedDeps"
