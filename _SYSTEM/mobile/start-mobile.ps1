$ErrorActionPreference = 'Stop'

$mobileRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = 'S:\all my projects\IG-AUTOMATIK'
$python = Join-Path $projectRoot '_SYSTEM\.venv-win\Scripts\python.exe'
$serverScript = Join-Path $mobileRoot 'server.py'
$out = Join-Path $mobileRoot 'mobile-server.out.log'
$err = Join-Path $mobileRoot 'mobile-server.err.log'

if (-not (Test-Path -LiteralPath $python)) {
    Write-Error "Python-Umgebung nicht gefunden: $python"
}
if (-not (Test-Path -LiteralPath $projectRoot)) {
    Write-Error "IG-AUTOMATIK-Projekt nicht gefunden: $projectRoot"
}

Set-Location -LiteralPath $mobileRoot
$quotedServerScript = '"' + $serverScript + '"'
$quotedRoot = '"' + $projectRoot + '"'
Start-Process -WindowStyle Hidden -FilePath $python `
    -ArgumentList @($quotedServerScript, '--project-root', $quotedRoot, '--host', '0.0.0.0', '--port', '8787') `
    -WorkingDirectory $mobileRoot `
    -RedirectStandardOutput $out `
    -RedirectStandardError $err | Out-Null
Write-Output 'IG-AUTOMATIK Mobile wurde unsichtbar im Hintergrund gestartet.'
