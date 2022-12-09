SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;

#create input table
DROP TABLE IF EXISTS input_table;
CREATE TABLE input_table as
	 SELECT hit,
	        atbat,
	        batter,
	        bc.game_id,
	        Date(local_date) as l_date
FROM batter_counts bc
JOIN game g ON g.game_id = bc.game_id
ORDER BY game_id ,l_date;

# rolling average calculation for game id = 12560
DROP TABLE IF EXISTS Rolling_Average;
CREATE TABLE Rolling_Average as
	SELECT sum(ifnull(it1.Hit,0))/nullif(sum(ifnull(it1.atbat,0)),0) as Rolling_average,
	       it.l_date,
	       it.game_id,
	       it.batter,
	       count(*) as cnt
# self joining input table
FROM input_table it JOIN input_table it1 ON it1.batter = it.batter AND it1.l_date BETWEEN DATE_ADD(it.l_date, interval -100 day)
AND DATE_SUB(it.l_date, INTERVAL 1 DAY) WHERE it.game_id = 12560
GROUP BY it.game_id ,it.batter, it.l_date;

