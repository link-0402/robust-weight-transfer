[CmdletBinding()]
# Packages the add-on as a flat-root Blender Extension ZIP (blender_manifest.toml
# and __init__.py at the archive root, per the Extensions platform's layout).
# This differs from build-release.ps1, which nests the add-on in a package-name
# folder for the legacy "Install from Disk" installer.
# The vendored dependencies must already be available in .\deps.
param(
    [string] $OutputPath = ''
)

$ErrorActionPreference = 'Stop'

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
$RepositoryRoot = (Resolve-Path (Join-Path $ScriptRoot '..')).Path
$ManifestFile = Join-Path $RepositoryRoot 'blender_manifest.toml'
$DependenciesDirectory = Join-Path $RepositoryRoot 'deps'

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $RepositoryRoot 'blender_repo\robust_weight_transfer.zip'
}
$ResolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)

if (-not (Test-Path -LiteralPath $ManifestFile -PathType Leaf)) {
    throw "Could not find the Blender extension manifest: $ManifestFile"
}
if (-not (Test-Path -LiteralPath $DependenciesDirectory -PathType Container)) {
    throw @"
The vendored dependencies directory is missing: $DependenciesDirectory
Install the release dependencies first, then run this script again.
"@
}

$RuntimeFiles = @(
    'blender_manifest.toml',
    '__init__.py',
    'weighttransfer.py',
    'util.py',
    'transfer.py',
    'seams.py',
    'third-logo-icon.png',
    'LICENSE',
    'README.md'
)

foreach ($File in $RuntimeFiles) {
    $SourcePath = Join-Path $RepositoryRoot $File
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
        throw "Required extension file is missing: $SourcePath"
    }
}

$OutputDirectory = Split-Path -Parent $ResolvedOutput
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
if (Test-Path -LiteralPath $ResolvedOutput) {
    Remove-Item -LiteralPath $ResolvedOutput -Force
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Archive = $null

try {
    $Archive = [System.IO.Compression.ZipArchive]::new(
        [System.IO.File]::Open($ResolvedOutput, [System.IO.FileMode]::Create),
        [System.IO.Compression.ZipArchiveMode]::Create)

    foreach ($File in $RuntimeFiles) {
        $SourcePath = Join-Path $RepositoryRoot $File
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive, $SourcePath, $File.Replace('\', '/'),
            [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }

    # Keep only runtime dependency files. __pycache__, bytecode, wheels, and test
    # data are not needed by Blender and can make the extension archive
    # unnecessarily large.
    $DependencyFiles = Get-ChildItem -LiteralPath $DependenciesDirectory -Recurse -File -Force |
        Where-Object {
            $_.FullName -notmatch '[\\/](__pycache__|tests)([\\/]|$)' -and
            $_.Extension -notin @('.pyc', '.pyo', '.whl')
        }

    foreach ($File in $DependencyFiles) {
        $RelativePath = 'deps/' + $File.FullName.Substring($DependenciesDirectory.Length).TrimStart([char]92, [char]47).Replace('\', '/')
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $Archive, $File.FullName, $RelativePath,
            [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
}
finally {
    if ($null -ne $Archive) {
        $Archive.Dispose()
    }
}

$ArchiveInfo = Get-Item -LiteralPath $ResolvedOutput
Write-Output ("Created {0} ({1:N1} MB)" -f $ArchiveInfo.FullName, ($ArchiveInfo.Length / 1MB))
