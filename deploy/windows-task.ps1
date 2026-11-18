# Planifie la collecte automatique sur un serveur (ou un PC) Windows : toutes les 30 minutes.
#
#   powershell -ExecutionPolicy Bypass -File deploy\windows-task.ps1            # installe
#   powershell -ExecutionPolicy Bypass -File deploy\windows-task.ps1 -Remove    # retire
#
# Prérequis : le venv existe déjà (lol-predictor\venv) avec les dépendances installées.
# Le journal est écrit dans lol-predictor\reports\refresh.log.
#
# Sur un SERVEUR, ajouter -Server : la tâche tourne alors même si personne n'est connecté
# (sans mot de passe enregistré ; à lancer depuis un compte administrateur).
param([switch]$Remove, [switch]$Server)

$ErrorActionPreference = "Stop"
$TaskName = "LoL predictor - collecte"
$App = Join-Path (Split-Path $PSScriptRoot -Parent) "lol-predictor"
$Python = Join-Path $App "venv\Scripts\python.exe"
$Log = Join-Path $App "reports\refresh.log"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Tâche retirée."
    exit 0
}
if (-not (Test-Path $Python)) {
    throw "venv introuvable : $Python. Crée-le d'abord (voir deploy\README.md)."
}
New-Item -ItemType Directory -Force (Split-Path $Log -Parent) | Out-Null

$Command = "set PYTHONIOENCODING=utf-8&& `"$Python`" -W ignore -m src.update.refresh_all >> `"$Log`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $Command" -WorkingDirectory $App
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes 30)
# Le relevé soloQ dure ~1 h 30 : limite large. Un passage déjà en cours n'est pas doublé.
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)

$Params = @{
    TaskName = $TaskName; Action = $Action; Trigger = $Trigger; Settings = $Settings; Force = $true
    Description = "Actualise les données LoL (Oracle, Leaguepedia, Polymarket, cotes, soloQ)"
}
if ($Server) {
    # S4U : s'exécute sans session ouverte et sans mot de passe stocké. Suffit ici, la collecte
    # ne fait que des requêtes web sortantes et écrit dans son propre dossier.
    # Le compte est lu sur le jeton Windows : dans une session SSH, USERDOMAIN n'est pas fiable.
    $Me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Params.Principal = New-ScheduledTaskPrincipal -UserId $Me -LogonType S4U
}
Register-ScheduledTask @Params | Out-Null

Write-Host "Tâche « $TaskName » installée : premier passage dans 2 minutes, puis toutes les 30 minutes."
Write-Host "  État des sources : cd `"$App`"; .\venv\Scripts\python.exe -m src.update.refresh_all --status"
Write-Host "  Journal          : $Log"
if (-not $Server) {
    Write-Host "La tâche ne tourne que lorsque ta session est ouverte. Sur un serveur, relance avec -Server."
}
