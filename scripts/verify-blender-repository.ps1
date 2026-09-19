[CmdletBinding()]
# Sanity-checks a generated Blender extension repository against the manifest:
# the index lists our extension id, the declared minimum Blender version
# matches, and the archive hash/size in the index match the actual file.
param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryPath,
    [string]$ExpectedMinimum = '5.2.0'
)

$ErrorActionPreference = 'Stop'
$ResolvedRepository = (Resolve-Path -LiteralPath $RepositoryPath).Path
$ArchivePath = Join-Path $ResolvedRepository 'robust_weight_transfer.zip'
$IndexPath = Join-Path $ResolvedRepository 'index.json'
if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    throw "Extension package not found: $ArchivePath"
}
if (-not (Test-Path -LiteralPath $IndexPath -PathType Leaf)) {
    throw "Repository index not found: $IndexPath"
}

$Entry = (Get-Content -LiteralPath $IndexPath -Raw | ConvertFrom-Json).data |
    Where-Object { $_.id -eq 'robust_weight_transfer' } |
    Select-Object -First 1
if ($null -eq $Entry) {
    throw 'Repository index does not contain robust_weight_transfer.'
}
if ($Entry.blender_version_min -ne $ExpectedMinimum) {
    throw "Blender minimum mismatch: expected $ExpectedMinimum, found $($Entry.blender_version_min)."
}

$Archive = Get-Item -LiteralPath $ArchivePath
if ([int64]$Entry.archive_size -ne $Archive.Length) {
    throw "Archive size mismatch: index=$($Entry.archive_size), actual=$($Archive.Length)."
}

$ActualHash = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if (([string]$Entry.archive_hash).ToLowerInvariant() -ne "sha256:$ActualHash") {
    throw "Archive hash mismatch: index=$($Entry.archive_hash), actual=sha256:$ActualHash."
}

Write-Host "Verified Blender extension repository: $($Archive.Length) bytes, sha256:$ActualHash"
