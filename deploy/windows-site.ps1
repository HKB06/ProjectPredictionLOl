# Fait tourner le site en permanence sur un serveur Windows, en accès PRIVÉ via Tailscale.
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows-site.ps1            # installe
#   powershell -ExecutionPolicy Bypass -File deploy\windows-site.ps1 -Remove    # retire
#
# Le site n'écoute que sur 127.0.0.1 : il n'est joignable ni depuis Internet ni depuis le
# réseau local. Tailscale le publie en HTTPS sur un port à part, pour les seuls appareils de
# ton réseau Tailscale. Une configuration « tailscale serve » déjà en place sur un autre port
# (un autre projet) n'est pas modifiée.
#
# Une tâche planifiée relance le site au démarrage du serveur et le surveille toutes les
# 5 minutes (s'il tourne déjà, rien ne se passe ; s'il s'est arrêté, il repart).
param([switch]$Remove, [int]$Port = 8501, [int]$TailscalePort = 8443)

$ErrorActionPreference = "Stop"
$TaskName = "LoL predictor - site"
$App = Join-Path (Split-Path $PSScriptRoot -Parent) "lol-predictor"
$Streamlit = Join-Path $App "venv\Scripts\streamlit.exe"
$Log = Join-Path $App "reports\site.log"
$Tailscale = "C:\Program Files\Tailscale\tailscale.exe"
$HasTailscale = Test-Path $Tailscale

if ($Remove) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    if ($HasTailscale) { & $Tailscale serve --https=$TailscalePort off | Out-Null }
    Write-Host "Site retiré (tâche supprimée, port Tailscale $TailscalePort libéré)."
    exit 0
}
if (-not (Test-Path $Streamlit)) { throw "venv introuvable : $Streamlit" }
New-Item -ItemType Directory -Force (Split-Path $Log -Parent) | Out-Null

# Adresse Tailscale du serveur : Streamlit doit la connaître pour accepter la connexion du navigateur.
$Dns = $null
if ($HasTailscale) {
    $Dns = ((& $Tailscale status --json | ConvertFrom-Json).Self.DNSName).TrimEnd(".")
}
$Extra = ""
if ($Dns) { $Extra = " --browser.serverAddress $Dns --browser.serverPort $TailscalePort" }

$Command = "set PYTHONIOENCODING=utf-8&& `"$Streamlit`" run app.py --server.address 127.0.0.1 " +
    "--server.port $Port --server.headless true$Extra >> `"$Log`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $Command" -WorkingDirectory $App
$Triggers = @(
    (New-ScheduledTaskTrigger -AtStartup),
    (New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5))
)
# Pas de limite de durée (le site tourne en continu) ; une seule instance à la fois.
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$Me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$Principal = New-ScheduledTaskPrincipal -UserId $Me -LogonType S4U
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Triggers -Settings $Settings `
    -Principal $Principal -Description "Site LoL predictor (Streamlit, local uniquement)" -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Tâche « $TaskName » installée et démarrée (http://127.0.0.1:$Port sur le serveur)."

if (-not $HasTailscale) {
    Write-Host "Tailscale absent : le site n'est accessible que depuis le serveur lui-même."
    exit 0
}
$Existing = (& $Tailscale serve status 2>&1 | Out-String)
if ($Existing -match ":$TailscalePort\b") {
    Write-Host "Le port Tailscale $TailscalePort est déjà utilisé : configuration laissée telle quelle."
    Write-Host "Relance avec -TailscalePort <autre port> si ce n'est pas ce site."
} else {
    & $Tailscale serve --bg --https=$TailscalePort "http://127.0.0.1:$Port" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "tailscale serve a échoué (code $LASTEXITCODE)" }
}
Write-Host "Adresse du site (appareils Tailscale uniquement) : https://${Dns}:$TailscalePort"
