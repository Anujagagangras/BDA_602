SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;

#calculation of annual average
DROP TABLE IF EXISTS Annual_Average;
CREATE TABLE Annual_Average as
select b.batter, YEAR(g.local_date),
    sum(Hit)/nullif(sum(atBat),0) as annual_avg
    FROM game g JOIN batter_counts b on g.game_id = b.game_id group by b.batter, year(g.local_date)
    order by b.batter,YEAR(g.local_date);

#calculation of historic average
DROP TABLE IF EXISTS Historic_Average;
CREATE TABLE Historic_Average as
select b.batter,
    sum(Hit)/nullif(sum(atBat),0) as Historic_avg
    FROM  batter_counts b group by b.batter
    order by  batter;

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

