USE baseball;
SHOW TABLES;

#calculation of annual average
DROP TABLE IF EXISTS Annual_Average;
CREATE TABLE Annual_Average as
select b.batter, YEAR(g.local_date),
    sum(Hit)/nullif(sum(atBat),0) as annual_avg
    FROM game g JOIN batter_counts b on g.game_id = b.game_id group by b.batter, year(g.local_date)
    order by  1,2;

#calculation of historic average
DROP TABLE IF EXISTS Historic_Average;
CREATE TABLE Historic_Average as
select b.batter,
    sum(Hit)/nullif(sum(atBat),0) as Historic_avg
    FROM  batter_counts b group by b.batter
    order by  1,2;


#calculation of rolling average
DROP TABLE IF EXISTS Rolling_Average;
CREATE TABLE Rolling_Average as
select b.batter,
    sum(Hit)/nullif(sum(atBat),0) as rolling_avg
    FROM game g JOIN batter_counts b on g.game_id = b.game_id
    where g.local_date>=Date_sub((select max(local_date) from game),interval 100 day)
    group by b.batter
    having sum(atBat)>0
    order by  1,2;



