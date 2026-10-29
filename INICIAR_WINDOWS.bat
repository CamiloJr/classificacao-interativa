@echo off
setlocal
cd /d "%~dp0"
title Classificacao Interativa - Aprendizado de Maquina
where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python
echo Python nao foi encontrado. Instale o Python 3.10 a 3.14 e marque "Add python.exe to PATH".
pause
exit /b 1
:use_py
set "PYCOMMAND=py -3"
goto prepare
:use_python
set "PYCOMMAND=python"
:prepare
if exist ".venv\Scripts\python.exe" goto dependencies
echo Criando ambiente virtual Python...
%PYCOMMAND% -m venv .venv
if errorlevel 1 goto error
:dependencies
.venv\Scripts\python.exe -c "import numpy, pygame" >nul 2>nul
if not errorlevel 1 goto start
echo Instalando dependencias. A primeira execucao precisa de acesso a internet...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto error
:start
echo Abrindo o programa...
.venv\Scripts\python.exe classificacao.py
if errorlevel 1 goto error
exit /b 0
:error
echo.
echo Nao foi possivel iniciar o aplicativo. Consulte o README.md.
pause
exit /b 1
