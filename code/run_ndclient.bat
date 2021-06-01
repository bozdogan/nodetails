@ECHO OFF
REM Quick utility to run NoDetails Client without installing

SETLOCAL
SET PYTHONPATH=%~dp0

python ndclient\app.py

ENDLOCAL