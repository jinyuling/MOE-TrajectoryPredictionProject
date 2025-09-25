@echo off
echo 启用Windows长路径支持
echo ======================

REM 检查是否以管理员权限运行
net session >nul 2>&1
if %errorLevel% == 0 (
    echo 已以管理员权限运行
) else (
    echo 错误: 需要以管理员权限运行此脚本
    echo 请右键点击此批处理文件，选择"以管理员身份运行"
    pause
    exit /b
)

REM 启用长路径支持
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f

if %errorLevel% == 0 (
    echo Windows长路径支持已成功启用
    echo 请重启计算机以使更改生效
) else (
    echo 启用长路径支持失败
)

pause