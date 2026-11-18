#!/usr/bin/env bash
# Installe la collecte automatique (et, en option, le site) sur un serveur Linux avec systemd.
#
#   bash deploy/install.sh            # collecte seule : données actualisées toutes les 30 min
#   bash deploy/install.sh --site     # + le site Streamlit, en service permanent (port 8501, local)
#
# À lancer depuis la racine du dépôt, avec l'utilisateur qui fera tourner les tâches (pas root).
# Relançable sans risque : ne touche ni aux données ni aux clés.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$REPO/lol-predictor"
USER_NAME="$(id -un)"
WITH_SITE=0
[[ "${1:-}" == "--site" ]] && WITH_SITE=1

if [[ "$USER_NAME" == "root" ]]; then
  echo "Ne lance pas ce script en root : utilise un compte normal (sudo sera demandé au besoin)." >&2
  exit 1
fi

PY="$(command -v python3.12 || command -v python3.13 || command -v python3.11 || command -v python3 || true)"
if [[ -z "$PY" ]] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11 ou plus récent est requis (Ubuntu 24.04 et Debian 12 conviennent)." >&2
  exit 1
fi
echo "==> Python : $("$PY" --version) · dépôt : $REPO · utilisateur : $USER_NAME"

echo "==> Environnement Python (venv + dépendances)"
if [[ ! -x "$APP/venv/bin/python" ]]; then
  "$PY" -m venv "$APP/venv" || { echo "Installe d'abord : sudo apt install python3-venv" >&2; exit 1; }
fi
"$APP/venv/bin/pip" install --quiet --upgrade pip
"$APP/venv/bin/pip" install --quiet -r "$APP/requirements.txt"

echo "==> Tâche planifiée : collecte toutes les 30 minutes"
sudo tee /etc/systemd/system/lol-refresh.service >/dev/null <<EOF
[Unit]
Description=LoL predictor - actualisation des données
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$APP
Environment=PYTHONIOENCODING=utf-8
ExecStart=$APP/venv/bin/python -W ignore -m src.update.refresh_all
# le relevé soloQ dure ~1 h 30 : on ne coupe pas avant
TimeoutStartSec=4h
Nice=10
EOF

sudo tee /etc/systemd/system/lol-refresh.timer >/dev/null <<EOF
[Unit]
Description=LoL predictor - collecte toutes les 30 minutes

[Timer]
OnBootSec=2min
OnUnitInactiveSec=30min
Persistent=true

[Install]
WantedBy=timers.target
EOF

if [[ "$WITH_SITE" == 1 ]]; then
  echo "==> Service : site Streamlit (écoute en local sur le port 8501)"
  sudo tee /etc/systemd/system/lol-site.service >/dev/null <<EOF
[Unit]
Description=LoL predictor - site Streamlit
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$APP
Environment=PYTHONIOENCODING=utf-8
ExecStart=$APP/venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
fi

sudo systemctl daemon-reload
sudo systemctl enable --now lol-refresh.timer
[[ "$WITH_SITE" == 1 ]] && sudo systemctl enable --now lol-site.service

echo
echo "Installé. Premier passage dans 2 minutes, puis toutes les 30 minutes."
echo "  État des sources   : cd $APP && venv/bin/python -m src.update.refresh_all --status"
echo "  Lancer maintenant  : sudo systemctl start lol-refresh.service"
echo "  Journal            : journalctl -u lol-refresh.service -n 50"
[[ "$WITH_SITE" == 1 ]] && echo "  Site               : http://127.0.0.1:8501 (voir deploy/README.md pour y accéder de l'extérieur)"
exit 0
