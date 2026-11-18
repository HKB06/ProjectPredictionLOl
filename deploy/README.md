# Mettre la collecte de données sur le serveur

But : que toutes les données soient actualisées en permanence, sans rien lancer à la main.
Une seule commande fait tout, et le serveur la relance toutes les 30 minutes :

```
python -m src.update.refresh_all
```

Chaque source a son rythme ; la commande ne refait que ce qui est dû. Une source en panne
n'arrête pas les autres et est retentée au passage suivant.

| Source | Rythme | Clé | Ce qu'elle apporte |
|---|---|---|---|
| `oracle` | 3 h | aucune (`google.key` en secours) | toutes les games et leurs stats (Oracle's Elixir) |
| `history` | une fois | aucune | saisons passées rejouées par le modèle (`data.history_years` dans `config.yaml`, 2023 à 2025) : CSV Oracle du même dossier Drive + coachs et lieux Leaguepedia |
| `leaguepedia` | 6 h | aucune | tournois, LAN/online, coachs, rosters, comptes soloQ |
| `polymarket` | 1 h | aucune | prix d'avant-match de chaque match terminé |
| `odds` | 2 h | `oddsapi.key` | cotes de clôture des bookmakers (odds-api.io) |
| `soloq` | 20 h | `riot.key` | rang soloQ et champions joués par les pros |

Sans clé, `odds` et `soloq` sont simplement ignorées ; le reste tourne.

## 1. Ce qu'il faut sur le serveur

- Python **3.11 ou plus récent** (Ubuntu 24.04 et Debian 12 conviennent ; Ubuntu 22.04 non).
- `git`, et un accès Internet sortant.
- Pour la collecte seule : n'importe quelle petite machine. Pour héberger aussi le site :
  prévoir au moins 2 Go de mémoire (il recalcule les modèles sur toute la saison).

## 2. Installation — Linux

```bash
sudo apt update && sudo apt install -y git python3-venv
git clone -b Seb https://github.com/HKB06/ProjectPredictionLOl.git
cd ProjectPredictionLOl
bash deploy/install.sh            # collecte seule
# ou : bash deploy/install.sh --site   (collecte + site sur le port 8501, en local)
```

Le script crée l'environnement Python, installe les dépendances et enregistre une tâche
`systemd` (`lol-refresh.timer`). Il est relançable sans risque.

## 2 bis. Installation — Windows

```powershell
git clone -b Seb https://github.com/HKB06/ProjectPredictionLOl.git
cd ProjectPredictionLOl\lol-predictor
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
cd ..
powershell -ExecutionPolicy Bypass -File deploy\windows-task.ps1 -Server
powershell -ExecutionPolicy Bypass -File deploy\windows-site.ps1      # optionnel : le site
```

`-Server` (depuis un compte administrateur) fait tourner la collecte même si personne n'est
connecté. Sans cette option, la tâche ne tourne que lorsque la session est ouverte.

`windows-site.ps1` lance le site en permanence sur `127.0.0.1:8501` (relancé au démarrage du
serveur, surveillé toutes les 5 minutes) et, si Tailscale est installé, le publie en HTTPS sur
le port 8443 pour les seuls appareils de ton réseau Tailscale :
`https://<nom-du-serveur>.<réseau>.ts.net:8443`. Une publication Tailscale déjà en place sur un
autre port n'est pas modifiée. Journal : `lol-predictor\reports\site.log`.

## 3. Les clés (à créer une fois, dans `lol-predictor/`)

Un fichier texte par clé, contenant uniquement la clé. Ces fichiers sont ignorés par git
(`*.key`) : ils ne partent jamais sur GitHub.

| Fichier | Où l'obtenir | Remarque |
|---|---|---|
| `oddsapi.key` | odds-api.io (la même clé que celle du site actuel) | 100 requêtes/heure |
| `riot.key` | developer.riotgames.com → « Register Product » → **Personal API Key** | La clé « Development » affichée par défaut expire toutes les 24 h : inutilisable sur un serveur. La clé Personal est gratuite, validée sous quelques jours. |
| `google.key` (optionnel) | console.cloud.google.com → activer « Google Drive API » → créer une clé API | Secours quand le lien public d'Oracle's Elixir répond « Too many users » |

Sous Linux : `chmod 600 lol-predictor/*.key`.

## 4. Vérifier que ça tourne

```bash
cd lol-predictor
venv/bin/python -m src.update.refresh_all --status     # état de chaque source
sudo systemctl start lol-refresh.service               # lancer un passage maintenant
journalctl -u lol-refresh.service -n 50                # journal
systemctl list-timers lol-refresh.timer                # prochain passage
```

Sous Windows : `.\venv\Scripts\python.exe -m src.update.refresh_all --status`, journal dans
`lol-predictor\reports\refresh.log`.

Le site affiche le même état : page **Modèles → Données**.

## 5. Le site sur un serveur Linux (option `--site`)

Le site écoute sur `127.0.0.1:8501` : il n'est **pas** exposé à Internet par défaut.

- Pour toi seul : tunnel SSH depuis ton PC, `ssh -L 8501:127.0.0.1:8501 utilisateur@serveur`,
  puis ouvrir http://localhost:8501.
- Pour l'ouvrir à d'autres : placer un reverse proxy HTTPS devant (Caddy ou nginx) et définir
  le mot de passe du site, `APP_PASSWORD = "..."` dans `lol-predictor/.streamlit/secrets.toml`
  (fichier ignoré par git). Sans `APP_PASSWORD`, le site est ouvert à tous : ne jamais ouvrir
  le port 8501 directement.

Le site lit les fichiers que la collecte met à jour : il se recalcule tout seul au premier
affichage qui suit une actualisation (environ une minute).

## 6. Mettre à jour le code

```bash
cd ProjectPredictionLOl
git pull --autostash origin Seb
lol-predictor/venv/bin/pip install -r lol-predictor/requirements.txt
sudo systemctl restart lol-site.service    # seulement si le site est installé
```

## Limites à connaître

- **Oracle's Elixir** est hébergé sur Google Drive, qui bloque régulièrement les
  téléchargements (« Too many users »). La collecte retente toutes les 30 minutes ; la clé
  `google.key` ajoute une voie de secours.
- **SoloQ** : Riot ne donne que l'état du moment. Le signal (LP gagnés, champions travaillés)
  se lit en comparant les relevés jour après jour : il n'existe qu'à partir du jour où la
  collecte démarre.
- Le serveur ne pousse rien sur GitHub. Les données collectées restent sur le serveur.
