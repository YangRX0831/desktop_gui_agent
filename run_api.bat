@echo off
setlocal
cd /d "%~dp0"
python -B main.py --model-mode api %*
