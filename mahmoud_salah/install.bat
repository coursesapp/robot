@echo off
title Robot Installer
echo ===================================================
echo   SETTING UP GRADUATION PROJECT ENVIRONMENT
echo ===================================================
echo.

:: 1. Create the Environment
echo Creating virtual environment (grad_env)...
call conda create -n grad_env python=3.10 -y

:: 2. Activate and Install
echo.
echo Installing Libraries (This may take a few minutes)...
call conda activate grad_env
pip install -r requirements.txt

echo.
echo ===================================================
echo   INSTALLATION COMPLETE!
echo ===================================================
echo You can now use 'run.bat' to start the robot.
pause