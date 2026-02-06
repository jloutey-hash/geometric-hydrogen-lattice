@echo off
REM Week 3: SciPy Convergence Analysis Runner
REM Run this to execute the full Week 3 validation

cd "c:\Users\jlout\OneDrive\Desktop\Model study\E8 Model\SU(2) model"

echo ========================================================================
echo WEEK 3: SciPy Convergence Analysis
echo ========================================================================
echo.
echo This will validate SU(2) lattice discretization against SciPy
echo Testing 6 modes at 5 resolutions (expected time: 5-10 minutes)
echo.

REM Step 1: Quick test
echo [Step 1/2] Running quick validation test...
echo.
python test_scipy_quick.py
if errorlevel 1 (
    echo ERROR: Quick test failed!
    pause
    exit /b 1
)

echo.
echo ========================================================================
echo Quick test passed! Starting full analysis...
echo ========================================================================
echo.

REM Step 2: Full analysis
echo [Step 2/2] Running full convergence analysis...
echo This may take 5-10 minutes...
echo.
python scipy_convergence.py

echo.
echo ========================================================================
echo WEEK 3 COMPLETE!
echo ========================================================================
echo.
echo Results saved to:
echo   - results\scipy_convergence.json (numerical data)
echo   - results\scipy_convergence.png  (convergence plots)
echo.
echo Next: Review results and update SU(2) paper
echo.
pause
