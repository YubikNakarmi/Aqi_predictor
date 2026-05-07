"""MySQL utilities for connecting to a MySQL database, creating tables, writing DataFrames, and executing queries, with error handling and logging."""

from modules import logging_utils
import pandas as pd
from sqlalchemy import create_engine, text, Column, String, MetaData, Table
from sqlalchemy.exc import SQLAlchemyError

logger = logging_utils.setup_logging(__name__)


class MySQLUtils:

    def __init__(self, mysql_uri: str):
        self.mysql_uri = mysql_uri
        self.engine = create_engine(mysql_uri, pool_pre_ping=True)

    def get_engine(self):
        return self.engine

    def create_table(self, columns: list, table_name: str):
        try:
            metadata = MetaData()
            table_columns = [Column(col, String(255)) for col in columns]
            table = Table(table_name, metadata, *table_columns)
            metadata.create_all(self.engine)
            logger.info(f"Table {table_name} created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise RuntimeError(f"Failed to create table {table_name}: {e}")

    def write_dataframe_to_mysql(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Write a DataFrame to MySQL using pandas to_sql with SQLAlchemy engine."""
        try:
            with self.engine.connect() as connection:
                df.to_sql(name=table_name, con=connection, if_exists=if_exists, index=False) #sqlalchemy ver >2.0
                connection.commit()
            logger.info(f"DataFrame successfully written to MySQL table {table_name}")
        except Exception as e:
            logger.error(f"Failed to write DataFrame to MySQL table {table_name}: {e}")
            raise RuntimeError(f"Failed to write DataFrame to MySQL table {table_name}: {e}")

    def extract_data(self, query: str) -> pd.DataFrame:
        """Extract data from MySQL using a SQL query."""
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection)
            logger.info(f"Data successfully extracted from MySQL with query: {query}")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Failed to extract data from MySQL with query {query}: {e}")
            raise RuntimeError(f"Failed to extract data from MySQL with query {query}: {e}")

    def execute_query(self, query: str):
        """Execute a raw SQL query (for INSERT, UPDATE, DELETE, etc.)."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text(query))
                connection.commit()
            logger.info(f"Query executed successfully: {query}")
        except SQLAlchemyError as e:
            logger.error(f"Failed to execute query: {e}")
            raise RuntimeError(f"Failed to execute query: {e}")

    def dispose(self):
        """Dispose of the engine connection pool."""
        self.engine.dispose()
        logger.info("Engine connection pool disposed")


