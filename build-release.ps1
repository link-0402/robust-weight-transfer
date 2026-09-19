[CmdletBinding(SupportsShouldProcess)]
# Creates a Blender legacy add-on ZIP with a single importable package at its root.
# The vendored dependencies must already be available in .\deps.
param(
    [string] $OutputDirectory,
    [ValidatePattern('^[A-Za-z_][A-Za-z0-9_]*$')]
    [string] $PackageName = 'robust_weight_transfer'
)

$ErrorActionPreference = 'Stop'

$RepositoryRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$InitFile = Join-Path $RepositoryRoot '__init__.py'
$DependenciesDirectory = Join-Path $RepositoryRoot 'deps'

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepositoryRoot 'dist'
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

if (-not (Test-Path -LiteralPath $InitFile -PathType Leaf)) {
    throw "Could not find the Blender extension metadata: $InitFile"
}

$InitSource = Get-Content -LiteralPath $InitFile -Raw
$VersionMatch = [regex]::Match(
    $InitSource,
    '"version"\s*:\s*\(\s*(?<version>[^)]*?)\s*\)'
)

if (-not $VersionMatch.Success) {
    throw "Could not read bl_info['version'] from $InitFile"
}

$VersionParts = $VersionMatch.Groups['version'].Value -split '\s*,\s*' |
    Where-Object { $_ -ne '' }
if ($VersionParts.Count -ne 3 -or ($VersionParts | Where-Object { $_ -notmatch '^\d+$' }).Count -gt 0) {
    throw "bl_info['version'] must contain three integer components in $InitFile"
}

$Version = $VersionParts -join '.'
$ArchiveName = "{0}-{1}.zip" -f $PackageName, $Version
$ArchivePath = Join-Path $OutputDirectory $ArchiveName

$RuntimeFiles = @(
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
        throw "Required release file is missing: $SourcePath"
    }
}

if (-not (Test-Path -LiteralPath $DependenciesDirectory -PathType Container)) {
    throw @"
The vendored dependencies directory is missing: $DependenciesDirectory
Install the release dependencies first, then run this script again.
"@
}

if ($WhatIfPreference) {
    Write-Output "What if: would create $ArchivePath"
    return
}

New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null

$StagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("rwt-release-" + [guid]::NewGuid().ToString('N'))
$StagingPackage = Join-Path $StagingRoot $PackageName

try {
    New-Item -ItemType Directory -Path $StagingPackage -Force | Out-Null

    foreach ($File in $RuntimeFiles) {
        Copy-Item -LiteralPath (Join-Path $RepositoryRoot $File) -Destination (Join-Path $StagingPackage $File)
    }

    # Keep only runtime dependency files. __pycache__, bytecode, wheels, and test
    # data are not needed by Blender and can make the release archive unnecessarily large.
    $StagingDependencies = Join-Path $StagingPackage 'deps'
    New-Item -ItemType Directory -Path $StagingDependencies -Force | Out-Null

    $DependencyFiles = Get-ChildItem -LiteralPath $DependenciesDirectory -Recurse -File -Force |
        Where-Object {
            $_.FullName -notmatch '[\\/](__pycache__|tests)([\\/]|$)' -and
            $_.Extension -notin @('.pyc', '.pyo', '.whl')
        }

    foreach ($File in $DependencyFiles) {
        $RelativePath = $File.FullName.Substring($DependenciesDirectory.Length).TrimStart([char]92, [char]47)
        $DestinationPath = Join-Path $StagingDependencies $RelativePath
        $DestinationDirectory = Split-Path -Parent $DestinationPath
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
        Copy-Item -LiteralPath $File.FullName -Destination $DestinationPath -Force
    }

    if (Test-Path -LiteralPath $ArchivePath) {
        if (-not $PSCmdlet.ShouldProcess($ArchivePath, 'replace existing release archive')) {
            return
        }
        Remove-Item -LiteralPath $ArchivePath -Force
    }

    if (-not $PSCmdlet.ShouldProcess($ArchivePath, 'create release archive')) {
        return
    }
    Compress-Archive -LiteralPath $StagingPackage -DestinationPath $ArchivePath -CompressionLevel Optimal

    $Archive = [System.IO.Compression.ZipFile]::OpenRead($ArchivePath)
    try {
        $ExpectedPrefix = "$PackageName/"
        # Compress-Archive uses backslashes in entry names on Windows, while
        # ZIP paths are conventionally written with forward slashes.
        $ArchiveNames = @($Archive.Entries | ForEach-Object {
            $_.FullName.Replace([char]92, [char]47)
        })
        $MissingEntries = @()
        if ($ArchiveNames -notcontains "${ExpectedPrefix}__init__.py") {
            $MissingEntries += "${ExpectedPrefix}__init__.py"
        }
        if (-not ($ArchiveNames | Where-Object { $_ -like "${ExpectedPrefix}deps/*" })) {
            $MissingEntries += "${ExpectedPrefix}deps/"
        }

        if ($MissingEntries.Count -gt 0) {
            throw "Release archive is missing expected entries: $($MissingEntries -join ', ')"
        }
    }
    finally {
        $Archive.Dispose()
    }

    $ArchiveInfo = Get-Item -LiteralPath $ArchivePath
    Write-Output ("Created {0} ({1:N1} MB)" -f $ArchiveInfo.FullName, ($ArchiveInfo.Length / 1MB))
}
finally {
    if (Test-Path -LiteralPath $StagingRoot) {
        Remove-Item -LiteralPath $StagingRoot -Recurse -Force
    }
}
