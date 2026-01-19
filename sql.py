import pymysql

conn = pymysql.connect(host='localhost', user='root', password='yubik123', db='mlflow_backend')

try:
    conn.ping(reconnect=True)
    print("active")
except:
    print("error")