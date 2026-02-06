@echo off
echo ============================================
echo Compiling LaTeX Manuscript to PDF
echo ============================================
echo.

REM Change to the directory containing this script
cd /d "%~dp0"

echo Step 1/2: First compilation pass...
pdflatex -interaction=nonstopmode manuscript.tex
echo.

echo Step 2/2: Second compilation pass (for cross-references)...
pdflatex -interaction=nonstopmode manuscript.tex
echo.

echo ============================================
echo Compilation complete!
echo ============================================
echo.

if exist manuscript.pdf (
    echo SUCCESS! PDF created: manuscript.pdf
    echo Opening PDF...
    start manuscript.pdf
) else (
    echo ERROR: PDF was not created. Check for errors above.
    pause
)

echo.
echo Cleaning up auxiliary files...
del manuscript.aux manuscript.log manuscript.out 2>nul

echo Done!
pause
