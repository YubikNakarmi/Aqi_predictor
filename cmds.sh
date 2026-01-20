docker compose build --pull --no-cache 

mlflow server --backend-store-uri sqlite:///db/mlflow.db \
--default-artifact-root wasbs://mlflow@yubikaqidata.blob.core.windows.net/ --port 5000

mlflow server --backend-store-uri pymysql://root:yubik123@localhost:3306/mlflow_backend \
--default-artifact-root wasbs://mlflow@yubikaqidata.blob.core.windows.net/ --port 5000
