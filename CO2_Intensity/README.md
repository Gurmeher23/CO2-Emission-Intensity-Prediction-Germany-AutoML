# CO2_Intensity
This repository contains the code for the CO2 Intensity project. This project aims to create a tool that can calculate regionalized generationa dn demand as well as electrciity geneartion and consumption CO2 intensity signals for Germany.

1. Create a virtual environment using conda or venv. 
2. Install cosema package locally using pip install -e .
3. The previous step should already install all dependencies. But you still need a solver for the optimization problem. You can install the solver using the following command:
```bash
conda install -c conda-forge glpk
```

or if you have a Gurobi license, you can install the Gurobi solver using the following command:
```bash
conda install -c gurobi gurobi
```
4. Create a keys.yaml file in the root directory with the following structure:
```yaml
# insert your api key for the ENTSO-E here
entsoe-key: YOUR_ENTSOE_API_KEY

# insert your api key for the BMRS API here
bmrs-key: YOUR_ENTSOE_API_KEY

influxdb:
  host: localhost
  database_name: cosema
  username: root
  password: root
```
5. Create a docker container with the influxdb and grafana image and dashboard. This software relies on an influxdb database, and therefore you need a docker installed on your machine. You can create the docker container using the following command:
```bash
docker compose up -d
```
6. You can select what steps to execute using the manual_runs.py file. This file is a script that runs the different steps of the project. 
7. In the folder "Schedulers" you can find the different schedulers used to run the different steps of the project.
8. This project is still in it's beta phase, so bugs and errors are expected. If you find any, please report them in the issues section of this repository.


