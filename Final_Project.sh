#!/bin/bash

sleep 10

if ! mysql -h mariadb -u root -e 'use baseball'; then
    echo "Baseball does not exists"
    mysql -h mariadb -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
    echo "Importing database tables"
    mysql -h mariadb -u root baseball < /baseball.sql
else
  echo "Baseball DOES exists"
fi

echo "Calling Final_Project.sql"
mysql -h mariadb -u root  baseball < ./Final_Project.sql 

mysql -h mariadb -u root baseball -e '
  SELECT * FROM rolling_final_joint ' > /results/dataset.csv

python3 ./Final_Project.py