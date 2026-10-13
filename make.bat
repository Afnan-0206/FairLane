@echo off
setlocal
if "%1"=="" goto test
if "%1"=="exp-noisy-neighbor" goto exp_noisy
if "%1"=="exp-heterogeneous" goto exp_hetero
if "%1"=="test" goto test
if "%1"=="clean" goto clean
goto unknown

:test
"C:\Program Files\Go\bin\go.exe" test -v ./policy
"C:\Program Files\Go\bin\go.exe" test -v ./sim
goto end

:exp_noisy
"C:\Program Files\Go\bin\go.exe" run ./cmd/exp_noisy_neighbor
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe ./analysis/plot_noisy_neighbor.py
    .venv\Scripts\python.exe ./analysis/plot_counters_over_time.py
) else (
    python ./analysis/plot_noisy_neighbor.py
    python ./analysis/plot_counters_over_time.py
)
goto end

:exp_hetero
"C:\Program Files\Go\bin\go.exe" run ./cmd/exp_heterogeneous
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe ./analysis/plot_heterogeneous.py
) else (
    python ./analysis/plot_heterogeneous.py
)
goto end

:clean
if exist experiments\results\*.json del /q experiments\results\*.json
if exist experiments\results\*.png del /q experiments\results\*.png
goto end

:unknown
echo make: *** No rule to make target '%1'.  Stop.
exit /b 1

:end
