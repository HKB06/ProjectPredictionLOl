@echo off
chcp 65001 >nul
cd /d "%~dp0lol-predictor"

rem --- Evite l'invite e-mail de Streamlit au 1er lancement (portable) ---
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo ============================================================
echo   0/2  Fermeture des anciennes instances Streamlit...
echo   (evite de servir du vieux code en memoire)
echo ============================================================
powershell -ExecutionPolicy Bypass -File kill_streamlit.ps1
echo.
echo ============================================================
echo   1/2  Mise a jour des donnees (Oracle's Elixir / Drive)...
echo ============================================================
venv\Scripts\python.exe -m src.update.download_data
echo.
echo ============================================================
echo   2/2  Lancement de l'application (matchs a venir)...
echo   (le navigateur va s'ouvrir ; laisse cette fenetre ouverte)
echo ============================================================
venv\Scripts\python.exe -m streamlit run app.py

pause
