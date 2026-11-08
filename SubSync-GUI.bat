@echo off
title SubSync Studio - Moteur Hybride Math + IA Locale
echo Lancement de SubSync Studio GUI (WSL2)...
wsl.exe -e bash -lc "cd /home/juliench/subsync && ./bin/subsync-gui"
pause
