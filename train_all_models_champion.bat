@echo off
echo ========================================
echo Training 5 Models - CHAMPION Level
echo 極致性能配置
echo ========================================
echo.
echo Configuration:
echo - Augmentation: CHAMPION (15 transforms)
echo - Epochs: 150
echo - Patience: 20
echo - Batch Size: 16
echo.
echo 預計總時間: 8-12小時
echo 目標: Val Dice ^> 0.6
echo ========================================
echo.
pause

echo.
echo [1/5] Training Model 1: BCE Loss
echo ----------------------------------------
python train_2d.py --model_id 1 --loss bce --aug_strength champion --epochs 150 --patience 20 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model 1 training failed!
    pause
    exit /b 1
)

echo.
echo [2/5] Training Model 2: Focal Loss
echo ----------------------------------------
python train_2d.py --model_id 2 --loss focal --aug_strength champion --epochs 150 --patience 20 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model 2 training failed!
    pause
    exit /b 1
)

echo.
echo [3/5] Training Model 3: Dice Loss
echo ----------------------------------------
python train_2d.py --model_id 3 --loss dice --aug_strength champion --epochs 150 --patience 20 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model 3 training failed!
    pause
    exit /b 1
)

echo.
echo [4/5] Training Model 4: Tversky Loss
echo ----------------------------------------
python train_2d.py --model_id 4 --loss tversky --aug_strength champion --epochs 150 --patience 20 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model 4 training failed!
    pause
    exit /b 1
)

echo.
echo [5/5] Training Model 5: Combo Loss
echo ----------------------------------------
python train_2d.py --model_id 5 --loss combo --aug_strength champion --epochs 150 --patience 20 --batch_size 16
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Model 5 training failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ALL 5 MODELS TRAINED SUCCESSFULLY!
echo ========================================
echo.
echo Next Steps:
echo 1. Check training history in checkpoints_2d_model^*/
echo 2. Implement TTA and Ensemble
echo 3. Evaluate on test set
echo.
pause
