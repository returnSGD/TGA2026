@echo off
setlocal EnableDelayedExpansion

set REPO_URL=https://gitee.com/game-for-all_0/TEngine.git
set TEMP_DIR=%TEMP%\TEngine_tmp_%RANDOM%
:: 脚本所在目录即工程根目录
set PROJECT_DIR=%~dp0

echo ========================================
echo  TEngine Updater
echo ========================================
echo.
echo Source : %REPO_URL%
echo Target : %PROJECT_DIR%
echo.

:: ── 路径合法性检查 ────────────────────────────────
echo [0/3] Checking project layout ...

:: 必须存在 UnityProject 子目录
if not exist "%PROJECT_DIR%UnityProject\" (
    echo [ERROR] 'UnityProject' not found under %PROJECT_DIR%
    echo         Please run this script from the TEngine project root.
    goto :fail
)

:: 必须存在目标 YooAsset 目录
if not exist "%PROJECT_DIR%UnityProject\Packages\YooAsset\" (
    echo [ERROR] 'UnityProject\Packages\YooAsset' not found.
    echo         Unexpected project structure.
    goto :fail
)

:: 必须存在目标 TEngine 目录
if not exist "%PROJECT_DIR%UnityProject\Assets\TEngine\" (
    echo [ERROR] 'UnityProject\Assets\TEngine' not found.
    echo         Unexpected project structure.
    goto :fail
)

:: git 可用性检查
where git >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'git' not found in PATH. Please install Git for Windows.
    goto :fail
)

echo [OK] Project layout verified.
echo.

:: ── 稀疏克隆，只拉取指定目录 ──────────────────────
echo [1/3] Cloning (sparse) ...
git clone --no-checkout --depth=1 --filter=blob:none "%REPO_URL%" "%TEMP_DIR%"
if errorlevel 1 (
    echo [ERROR] git clone failed.
    goto :cleanup
)

cd /d "%TEMP_DIR%"

git sparse-checkout init --cone
git sparse-checkout set UnityProject/Packages/YooAsset UnityProject/Assets/TEngine
git checkout
if errorlevel 1 (
    echo [ERROR] git checkout failed.
    goto :cleanup
)

echo.
echo [2/3] Syncing UnityProject\Packages\YooAsset ...
robocopy "%TEMP_DIR%\UnityProject\Packages\YooAsset" ^
         "%PROJECT_DIR%UnityProject\Packages\YooAsset" ^
         /MIR /NFL /NDL /NJH /NJS /NC /NS
if errorlevel 8 (
    echo [ERROR] robocopy failed for YooAsset.
    goto :cleanup
)

echo.
echo [3/3] Syncing UnityProject\Assets\TEngine ...
robocopy "%TEMP_DIR%\UnityProject\Assets\TEngine" ^
         "%PROJECT_DIR%UnityProject\Assets\TEngine" ^
         /MIR /NFL /NDL /NJH /NJS /NC /NS
if errorlevel 8 (
    echo [ERROR] robocopy failed for TEngine.
    goto :cleanup
)

echo.
echo [OK] Update complete!

:cleanup
cd /d "%PROJECT_DIR%"
if exist "%TEMP_DIR%" rmdir /S /Q "%TEMP_DIR%"
goto :done

:fail
echo.
echo [ABORTED] No changes were made.

:done
echo.
pause
