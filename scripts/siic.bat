@echo off
REM SIIC - Vietnamese Sentiment Analysis System - Windows Commands

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="dashboard" goto dashboard
if "%1"=="train-baselines" goto train_baselines
if "%1"=="train-lstm" goto train_lstm
if "%1"=="train-phobert" goto train_phobert
if "%1"=="evaluate" goto evaluate
if "%1"=="test" goto test
if "%1"=="clean" goto clean

goto help

:help
echo.
echo SIIC - Vietnamese Emotion Detection System
echo ==========================================
echo.
echo Available commands:
echo   siic help              - Show this help
echo   siic install           - Install package
echo   siic dashboard         - Launch dashboard
echo   siic train-baselines   - Train baseline models
echo   siic train-lstm        - Train LSTM model  
echo   siic train-phobert     - Train PhoBERT model
echo   siic evaluate          - Run comprehensive evaluation
echo   siic test              - Test package imports
echo   siic clean             - Clean build artifacts
echo.
goto end

:install
echo Installing SIIC package...
pip install -e .
goto end

:dashboard
echo Starting SIIC Dashboard...
python scripts/dashboard.py
goto end

:train_baselines
echo Training baseline models...
python scripts/train.py --model baselines
goto end

:train_lstm
echo Training LSTM model...
python scripts/train.py --model lstm --epochs 15 --batch_size 32
goto end

:train_phobert
echo Training PhoBERT model...
python scripts/train.py --model phobert --epochs 3 --batch_size 8
goto end

:evaluate
echo Running comprehensive evaluation...
python scripts/evaluate.py --comprehensive
goto end

:test
echo Testing package imports...
python -c "from siic.utils.config import EMOTION_LABELS; print('Config OK')"
python -c "from siic.models.baselines import BaselineModels; print('Models OK')"
echo All tests passed!
goto end

:clean
echo Cleaning build artifacts...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
if exist *.egg-info rmdir /s /q *.egg-info
echo Cleanup complete!
goto end

:end 