import pymysql
from modules import logging_utils
import pandas as pd


logger = logging_utils.setup_logging(__name__)

class MySQLUtils:

    mysql_uri: str

    def __init__(self, mysql_uri: str):
        self.mysql_uri = mysql_uri

    @staticmethod
    def connect_to_mysql(mysql_uri):
        try:
            connection = pymysql.connect(mysql_uri)
            logger.info(f"Successfully connected to MySQL database at {mysql_uri}")
            return connection
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database at {mysql_uri}: {e}")
            raise ConnectionError(f"Failed to connect to MySQL database at {mysql_uri}: {e}")


    def create_table(self,rows: list, table_name: str):
        connection = self.connect_to_mysql(self.mysql_uri)
        try:
            with connection.cursor() as cursor:
                # Create table with list columns
                columns = ', '.join([f"{col} VARCHAR(255)" for col in rows]) #concat string with commas
                sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})"
                cursor.execute(sql)
            connection.commit()
            logger.info(f"Table {table_name} created successfully")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise RuntimeError(f"Failed to create table {table_name}: {e}")
        finally:
            connection.close()

        
    def write_dataframe_to_mysql(self, df, table_name):
        connection = self.connect_to_mysql(self.mysql_uri)
        try:
            with connection.cursor() as cursor:
                for _, row in df.iterrows():
                    placeholders = ', '.join(['%s'] * len(row))#concat with number of rows
                    sql = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES ({placeholders})"
                    cursor.execute(sql, tuple(row))
            connection.commit()
            logger.info(f"DataFrame successfully written to MySQL table {table_name}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to MySQL table {table_name}: {e}")
            raise RuntimeError(f"Failed to write DataFrame to MySQL table {table_name}: {e}")
        finally:
            connection.close()

    def extract_data(self, query: str) -> pd.DataFrame:
        connection = self.connect_to_mysql(self.mysql_uri)
        try:
            df = pd.read_sql(query, connection)
            logger.info(f"Data successfully extracted from MySQL with query: {query}")
            return df
        except Exception as e:
            logger.error(f"Failed to extract data from MySQL with query {query}: {e}")
            raise RuntimeError(f"Failed to extract data from MySQL with query {query}: {e}")
        finally:
            connection.close()