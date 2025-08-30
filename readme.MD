# AQI predictor with Airflow

This project uses historical and live data to predict AQI of Shankapark station in Kathmandu. 
## Configurations

### Building docker images

Build all the docker images first for running the scripts 
``` sh
docker build -t data_train:v1 -f scripts train.dockerfile scripts\train.dockerfile
```
### Configuring docker-copose

## Running
Run the following command on root directory
``` sh 
docker compose up
```


