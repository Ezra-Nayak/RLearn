@echo off
title CrossyRoad SIM Trainer

:: Change directory to your RLearn project folder
cd /d "C:\Users\User\PycharmProjects\PythonProject\RLearn"

:: Activate the shared virtual environment
call "C:\Users\User\PycharmProjects\PythonProject\CrossyLearn_v2\.venv\Scripts\activate.bat"

:: Start training
echo [SYSTEM] Starting Trainer...
python train_ppo_sim.py

pause
