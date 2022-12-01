
sleep 15

if ! mysql -h mariadb -u root -e 'use baseball'; then
    echo "Baseball DOES NOT exists"
    mysql -h mariadb -u root -e "CREATE DATABASE IF NOT EXISTS baseball"
    echo "Importing database tables"
    mysql -h mariadb -u root baseball < /scripts/baseball.sql
else
  echo "Baseball DOES exists"
fi

echo "Calling Homework_6.sql"
mysql -h mariadb -u root  baseball < /scripts/Homework_6.sql

mysql -h my-db -u root baseball -e '
  SELECT * FROM Rolling_Average where game_id=12560;' > /results/output.txt