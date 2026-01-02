# AQI predictor with Airflow

This project uses historical and live data to predict AQI of Shankapark station in Kathmandu. All the data is stored is csv file no exteral database is used. 

## Setting up

### DVC setup

Dvc is setup with azure blob storage
``` sh
dvc pull 
dvc get
```


### Building docker images

Build all the docker images first for running the ML scripts 
``` sh
docker build -t data_train:v1 -f scripts train.dockerfile .
```

``` sh
docker build -t data_predict:v1 -f scripts predict.dockerfile scripts\predict.dockerfile
```
### Configuring docker-compose
On data_train_dag.py setup the location of root directory. 
```python
host_path = "D:/pypipeline/data"
```

## Running
Run the following command on root directory
``` sh 
docker compose up
```


