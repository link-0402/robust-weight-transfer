[CmdletBinding()]
# Packages the extension and generates a static Blender Extensions repository
# (index.json + index.html) that Blender can add as a Remote Repository, or
# that users can install from directly via drag-and-drop / the web install link.
param(
    [string]$RepositoryPath = '',
    [string]$BlenderPath = ''
)

$ErrorActionPreference = 'Stop'

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = Join-Path $ScriptRoot '..\blender_repo'
}

$ResolvedRepository = [System.IO.Path]::GetFullPath($RepositoryPath)
$PackageScript = Join-Path $ScriptRoot 'package-blender-extension.ps1'
$PackagePath = Join-Path $ResolvedRepository 'robust_weight_transfer.zip'

$BlenderCommand = if ([string]::IsNullOrWhiteSpace($BlenderPath)) {
    Get-Command blender -ErrorAction SilentlyContinue
} else {
    $ResolvedBlenderPath = [System.IO.Path]::GetFullPath($BlenderPath)
    if (-not (Test-Path -LiteralPath $ResolvedBlenderPath -PathType Leaf)) {
        throw "Blender executable not found: $ResolvedBlenderPath"
    }
    Get-Item -LiteralPath $ResolvedBlenderPath
}
if ($null -eq $BlenderCommand) {
    throw 'Blender 4.2 or newer is required on PATH, or pass -BlenderPath.'
}
$BlenderExecutable = if ($BlenderCommand.PSObject.Properties.Name -contains 'Source') {
    $BlenderCommand.Source
} else {
    $BlenderCommand.FullName
}

New-Item -ItemType Directory -Path $ResolvedRepository -Force | Out-Null
& $PackageScript -OutputPath $PackagePath
if (-not (Test-Path -LiteralPath $PackagePath -PathType Leaf)) {
    throw "Extension package creation did not produce the expected archive: $PackagePath"
}

& $BlenderExecutable --background --command extension server-generate "--repo-dir=$ResolvedRepository" --html
if ($LASTEXITCODE -ne 0) {
    throw "Blender extension repository generation failed with exit code $LASTEXITCODE."
}

Write-Output "Generated Blender extension repository at $ResolvedRepository"
