SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;

#calculation of rolling average
DROP TABLE IF EXISTS Input_Table;  ## input table to calculate rolling average
CREATE TABLE Input_Table as
    select b.game_id,
           b.batter,
           g.local_date,
           b.Hit,
           b.atBat
FROM game g JOIN batter_counts b on g.game_id = b.game_id
having atBat>0;

DROP TABLE IF EXISTS Rolling_Average;
CREATE TABLE Rolling_Average as
SELECT *,
       AVG(Hit/atBat) OVER(
           PARTITION BY batter
           ORDER BY local_date
           ROWS BETWEEN 99 PRECEDING AND CURRENT ROW
           )
        AS 100_days_rolling_average
FROM Input_Table;

