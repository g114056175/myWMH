@echo off
REM nnU-Net环境变量设置
REM 在运行任何nnU-Net命令前必须先运行此脚本

echo ========================================
echo 设置nnU-Net环境变量
echo ========================================

set nnUNet_raw=d:\VSCode\AIOT_E1\nnUnet\nnUNet_raw
set nnUNet_preprocessed=d:\VSCode\AIOT_E1\nnUnet\nnUNet_preprocessed
set nnUNet_results=d:\VSCode\AIOT_E1\nnUnet\nnUNet_results

echo.
echo 环境变量已设置:
echo   nnUNet_raw=%nnUNet_raw%
echo   nnUNet_preprocessed=%nnUNet_preprocessed%
echo   nnUNet_results=%nnUNet_results%
echo.
echo ✓ 完成
echo.
echo 提示: 此设置仅在当前终端有效
echo       每次新开终端都需要重新运行此脚本
