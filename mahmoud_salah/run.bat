@echo off
title AI Robot
echo Starting the Robot...

:: 1. Activate the environment we created
call conda activate grad_env

:: 2. Run the main script
python main_robot.py

:: 3. Keep window open if it crashes so he can see why
pause