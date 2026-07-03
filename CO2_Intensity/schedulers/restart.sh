#!/bin/sh
pkill -f "python schedulers/initial_calculations.py"
pkill -f "python schedulers/updated_calculations.py"
pkill -f "python schedulers/forecast_calculations.py"
sleep 10
nohup python schedulers/initial_calculations.py &
nohup python schedulers/updated_calculations.py &
nohup python schedulers/forecast_calculations.py &
disown -a