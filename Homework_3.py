# import os
# import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

# Making spark object
spark = SparkSession.builder.master("local[*]").getOrCreate()

# creating connection to Mariadb
database_name = "baseball"
port = "3306"
user = "root"
password = " "

# reading data from jdbc and creating input table
Input_Table = (
    spark.read.format("jdbc")
    .options(
        url=f"jdbc:mysql://localhost:{port}/{database_name}",
        driver="com.mysql.cj.jdbc.Driver",
        dbtable="(select b.game_id,b.batter,DATE(g.local_date) as local_date,b.Hit,b.AtBat \
                FROM game g JOIN batter_counts b on g.game_id = b.game_id \
                having atBat>0)batters",
        user=user,
        password=password,
    )
    .load()
)

# create or replace input table to calculate rolling average
Input_Table.createOrReplaceTempView("Input_Table")
Input_Table.show()

Input_Table.persist(StorageLevel.DISK_ONLY)

rollingAvg = spark.sql(
    """SELECT *, AVG(Hit/atBat) OVER(
                                    PARTITION BY batter
                                    ORDER BY local_date
                                    ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
                                    )
                                AS 100_days_rolling_average
                                FROM Input_Table"""
)
rollingAvg.show()
