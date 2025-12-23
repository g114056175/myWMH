"""
訓練5個模型的批次腳本
"""
@echo off
echo ========================================
echo Training 5 2D Models - PGS Style
echo ========================================

echo.
echo Model 1: BCE Loss
python train_2d.py --model_id 1 --loss bce --batch_size 16 --epochs 100

echo.
echo Model 2: Focal Loss
python train_2d.py --model_id 2 --loss focal --batch_size 16 --epochs 100

echo.
echo Model 3: Dice Loss
python train_2d.py --model_id 3 --loss dice --batch_size 16 --epochs 100

echo.
echo Model 4: Tversky Loss
python train_2d.py --model_id 4 --loss tversky --batch_size 16 --epochs 100

echo.
echo Model 5: Combo Loss
python train_2d.py --model_id 5 --loss combo --batch_size 16 --epochs 100

echo.
echo ========================================
echo All models trained!
echo ========================================
pause
