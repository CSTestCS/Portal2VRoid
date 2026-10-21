@echo off
REM Build Portal2VRoid.exe locally (needs Python 3.10+ on Windows).
cd /d "%~dp0\.."
python -m pip install -r requirements.txt pyinstaller || exit /b 1
python -m PyInstaller --noconfirm --clean --onefile --windowed --name Portal2VRoid ^
  --add-data "portal2vroid/game_files;portal2vroid/game_files" packaging/entry.py || exit /b 1
python -m PyInstaller --noconfirm --onefile --console --name Portal2VRoid-cli ^
  --add-data "portal2vroid/game_files;portal2vroid/game_files" packaging/entry.py || exit /b 1
echo.
echo Built dist\Portal2VRoid.exe and dist\Portal2VRoid-cli.exe
