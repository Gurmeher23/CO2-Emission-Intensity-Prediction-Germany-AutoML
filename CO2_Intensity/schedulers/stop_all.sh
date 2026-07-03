#!/bin/sh
pkill -f "python schedulers/updated_calculations.py"
pkill -f "python schedulers/initial_calculations.py"
pkill -f "python schedulers/forecast_calculations.py"