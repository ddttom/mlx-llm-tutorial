@echo off
REM Script to remove large files from Git history
REM Created for MLX LLM Tutorial project

echo === Git Large File Removal Tool ===
echo This script will help remove the Miniconda installer from your Git history
echo Make sure you have committed all your changes before proceeding
echo.

REM Check if git-filter-repo is installed
git filter-repo --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo git-filter-repo is not installed.
    echo Would you like to install it now? (y/n)
    set /p install_choice=
    
    if /i "%install_choice%"=="y" (
        echo Installing git-filter-repo...
        pip install git-filter-repo
        if %ERRORLEVEL% NEQ 0 (
            echo Failed to install git-filter-repo with pip.
            echo Please install git-filter-repo manually and run this script again.
            echo You can install it with: pip install git-filter-repo
            exit /b 1
        )
    ) else (
        echo Please install git-filter-repo manually and run this script again.
        echo You can install it with: pip install git-filter-repo
        exit /b 1
    )
)

REM Check if we're in a git repository
git rev-parse --is-inside-work-tree >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Not in a git repository.
    echo Please run this script from within your git repository.
    exit /b 1
)

REM Confirm with the user
echo This will permanently remove the Miniconda installer file from your Git history.
echo WARNING: This operation is destructive and cannot be undone!
echo It will rewrite your Git history, which will require a force push.
echo Are you sure you want to continue? (y/n)
set /p choice=

if /i NOT "%choice%"=="y" (
    echo Operation cancelled.
    exit /b 0
)

REM Create a backup branch just in case
for /f "tokens=*" %%a in ('git symbolic-ref --short HEAD') do set current_branch=%%a
for /f "tokens=*" %%a in ('powershell -Command "Get-Date -Format 'yyyyMMddHHmmss'"') do set timestamp=%%a
set backup_branch=%current_branch%_backup_%timestamp%
echo Creating backup branch: %backup_branch%
git branch "%backup_branch%"

REM Remove the large file from history
echo Removing Miniconda3-latest-MacOSX-arm64.sh from Git history...
echo This may take a while...
git filter-repo --path Miniconda3-latest-MacOSX-arm64.sh --invert-paths

REM Verify the file is gone
echo File removal complete!
echo Checking if the file is still in the repository...
git ls-files | findstr "Miniconda3-latest-MacOSX-arm64.sh" >nul
if %ERRORLEVEL% EQU 0 (
    echo Warning: The file is still in your working directory.
    echo It has been removed from history but is still present locally.
    echo You should delete it manually and commit the change.
) else (
    echo The file is no longer tracked by Git.
)

REM Instructions for pushing
echo.
echo === Next Steps ===
echo To push these changes to GitHub, you'll need to force push:
echo git push origin %current_branch% --force
echo Note: This will overwrite the remote history. Make sure your team is aware of this change.
echo If anything went wrong, you can restore from the backup branch:
echo git checkout %backup_branch%
echo git branch -D %current_branch%
echo git checkout -b %current_branch%

echo.
echo === Preventing Future Issues ===
echo 1. The .gitignore file has been updated to exclude installer files
echo 2. Always use the dedicated ~/ai-training directory for your projects
echo 3. Use the one-step installation method for Miniconda:
echo    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh ^| bash

echo.
echo Done!
pause