
USE baseball;

ALTER TABLE inning MODIFY COLUMN game_id INT UNSIGNED NOT NULL;

## one game has impossible data for the temperature, not taking this game to consideration
DELETE FROM boxscore WHERE temp ='7882 degrees';

## response variable from boxscore table
DROP TABLE IF EXISTS game_info;
CREATE TABLE game_info
SELECT game_id,
       substr(temp,1, instr(temp, ' ') -1) AS temp,
       away_runs,
       away_errors,
       home_runs,
       home_errors,
       CASE WHEN winner_home_or_away="H" THEN 1 ELSE 0 END AS HomeTeamWins
FROM boxscore;

CREATE INDEX game_info_index ON game_info (game_id);

## Select columns needed from game table
DROP TABLE IF EXISTS game_date;
CREATE TABLE game_date
SELECT game_id,
       DATE(local_date) AS local_date
FROM game;

CREATE INDEX game_date_index ON game_date (game_id);

## Select columns needed from team_pitching_counts [Home Team]
DROP TABLE IF EXISTS ht_pitching_data;
CREATE TABLE ht_pitching_data
SELECT t.game_id,
       g.local_date,
       t.team_id AS home_team,
       t.plateApperance AS home_PA,
       (t.Hit / t.atBat) AS home_BA,
       t.Hit AS home_H,
       t.Home_Run AS home_HR,
       (t.Home_Run / 9) AS home_HR9,
       t.Walk AS home_BB,
       (t.Walk / 9) AS home_BB9,
       t.Strikeout AS home_K,
       (t.Strikeout / 9) AS home_K9,
       (t.Strikeout / NULLIF(t.Walk,0)) AS home_KBB,
       t.Triple_Play AS home_TP,
       t.Flyout AS home_Flyout,
       t.Grounded_Into_DP AS home_GIDP,
    SUM(t.Strikeout) AS Strikeouts,
	SUM(t.plateApperance) as PlateApperance,
	SUM(t.Single) as Single,
	SUM(t.`Double`) as Double_,
	SUM(t.Triple) as Triple
FROM team_pitching_counts t
JOIN game_date g
ON t.game_id = g.game_id
WHERE homeTeam = 1
GROUP BY g.game_id,
         g.local_date,
         t.team_id;

CREATE UNIQUE INDEX ht_pitching_data_index
ON ht_pitching_data (game_id, home_team);
CREATE INDEX ht_id_index ON ht_pitching_data (game_id);

# home batting data from team batting counts
DROP TABLE IF EXISTS ht_batting_data;
CREATE TABLE ht_batting_data
SELECT TBC.game_id,
       TBC.team_id,
    COALESCE(ROUND(SUM(TBC.toBase) /SUM(NULLIF(TBC.atBat,0)),2),0) as Slugging_percentage,
	COALESCE(ROUND(SUM(TBC.Hit)/sum(NULLIF(TBC.atBat,0)),2),0) as Batting_Average,
	COALESCE(ROUND(sum(TBC.Walk) / SUM(NULLIF(TBC.Strikeout,0)),2),0) as Walk_strikeout_ratio,
	COALESCE(ROUND(SUM(TBC.Ground_Out) /SUM(NULLIF(TBC.Fly_Out,0)),2),0) as Ground_fly_ball_ratio,
	SUM(TBC.Intent_Walk) as Intentional_Walk,
    COALESCE(ROUND(SUM(TBC.atBat)/SUM(NULLIF(TBC.Home_Run,0)),2),0) as At_bats_per_home_run,
    COALESCE(ROUND(SUM(TBC.Home_Run) /SUM(NULLIF(TBC.Hit,0)),2),0) as Home_runs_per_hit
FROM game_date gd
LEFT JOIN  team_batting_counts as TBC
on TBC.game_id = gd.game_id
WHERE homeTeam = 1
GROUP BY gd.game_id,
         gd.local_date,
         TBC.team_id;
CREATE UNIQUE INDEX ht_batting_data_index
ON ht_batting_data (game_id, team_id);
CREATE INDEX ht_id_index ON ht_batting_data (game_id);

# away batting data from team batting counts
DROP TABLE IF EXISTS at_batting_data;
CREATE TABLE at_batting_data
SELECT TBC.game_id,
       TBC.team_id,
    COALESCE(ROUND(SUM(TBC.toBase) /SUM(NULLIF(TBC.atBat,0)),2),0) as Slugging_percentage,
	COALESCE(ROUND(SUM(TBC.Hit)/sum(NULLIF(TBC.atBat,0)),2),0) as Batting_Average,
	COALESCE(ROUND(sum(TBC.Walk) / SUM(NULLIF(TBC.Strikeout,0)),2),0) as Walk_strikeout_ratio,
	COALESCE(ROUND(SUM(TBC.Ground_Out) /SUM(NULLIF(TBC.Fly_Out,0)),2),0) as Ground_fly_ball_ratio,
	SUM(TBC.Intent_Walk) as Intentional_Walk,
    COALESCE(ROUND(SUM(TBC.atBat)/SUM(NULLIF(TBC.Home_Run,0)),2),0) as At_bats_per_home_run,
    COALESCE(ROUND(SUM(TBC.Home_Run) /SUM(NULLIF(TBC.Hit,0)),2),0) as Home_runs_per_hit
FROM game_date gd
LEFT JOIN team_batting_counts as TBC
on TBC.game_id = gd.game_id
WHERE awayTeam = 1
GROUP BY gd.game_id,
         gd.local_date,
         TBC.team_id;

CREATE UNIQUE INDEX at_batting_data_index
ON at_batting_data (game_id, team_id);
CREATE INDEX at_id_index ON at_batting_data (game_id);

## Select columns needed from team_pitching_counts [Away Team]
DROP TABLE IF EXISTS at_pitching_data;
CREATE TABLE at_pitching_data
SELECT t.game_id,
       g.local_date,
       t.team_id AS away_team,
       t.plateApperance AS away_PA,
       (t.Hit / t.atBat) AS away_BA,
       t.Hit AS away_H,
       t.Home_Run AS away_HR,
       (t.Home_Run / 9) AS away_HR9,
       t.Walk AS away_BB,
       (t.Walk / 9) AS away_BB9,
       t.Strikeout AS away_K,
       (t.Strikeout / 9) AS away_K9,
       (t.Strikeout / NULLIF(t.Walk,0)) AS away_KBB,
       t.Triple_Play AS away_TP,
       t.Flyout AS away_Flyout,
       t.Grounded_Into_DP AS away_GIDP,
       SUM(t.Strikeout) AS Strikeouts,
	   SUM(t.plateApperance) as PlateApperance,
	   SUM(t.Single) as Single,
	   SUM(t.`Double`) as Double_,
	   SUM(t.Triple) as Triple
FROM team_pitching_counts t
JOIN game_date g
ON t.game_id = g.game_id
WHERE awayTeam = 1
GROUP BY g.game_id,
         g.local_date,
         t.team_id;

CREATE UNIQUE INDEX at_pitching_data_index
ON at_pitching_data (game_id, away_team);
CREATE INDEX at_id_index ON at_pitching_data (game_id);


## join home/away pitching team table
DROP TABLE IF EXISTS ha_pitching_joint;
CREATE TABLE ha_pitching_joint
SELECT h.game_id,
       h.local_date,
       h.home_team,
       a.away_team,
       h.home_PA,
       h.home_BA,
       h.home_H,
       h.home_HR,
       h.home_HR9,
       h.home_BB,
       h.home_BB9,
       h.home_K,
       h.home_K9,
       h.home_KBB,
       h.home_TP,
       h.home_Flyout,
       h.home_GIDP,
       i.home_runs,
       i.home_errors,
       a.away_PA,
       a.away_BA,
       a.away_H,
       a.away_HR,
       a.away_HR9,
       a.away_BB,
       a.away_BB9,
       a.away_K,
       a.away_K9,
       a.away_KBB,
       a.away_TP,
       a.away_Flyout,
       a.away_GIDP,
       i.away_runs,
       i.away_errors
FROM ht_pitching_data h
JOIN at_pitching_data a
ON h.game_id = a.game_id
JOIN game_info i
ON h.game_id = i.game_id

ORDER BY h.game_id;

CREATE INDEX ha_id_index ON ha_pitching_joint (game_id);
CREATE INDEX ha_d_index ON ha_pitching_joint (local_date);


## join home/away batting team table
DROP TABLE IF EXISTS ha_batting_joint;
CREATE TABLE ha_batting_joint
SELECT HT.game_id,
       HT.team_id as HomeTeamID,
       HT.Slugging_percentage as Home_Team_Slugging_Percentage,
       AT.Slugging_percentage as Away_Team_Slugging_Percentage,
       HT.Batting_Average as Home_Team_Batting_Average,
       AT.Batting_Average as Away_Team_Batting_Average,
       HT.Walk_strikeout_ratio as Home_Team_Walk_strikeout_ratio,
       AT.Walk_strikeout_ratio as Away_Team_Walk_strikeout_ratio,
       HT.Ground_fly_ball_ratio as Home_Team_Ground_fly_ball_ratio,
       AT.Ground_fly_ball_ratio as Away_Team_Ground_fly_ball_ratio,
       HT.Intentional_Walk as Home_Team_Intentional_Walk,
       AT.Intentional_Walk as Away_Team_Intentional_Walk,
       HT.At_bats_per_home_run as Home_Team_At_bats_per_home_run,
       AT.At_bats_per_home_run as Away_Team_At_bats_per_home_run,
       HT.Home_runs_per_hit as Home_Team_Home_runs_per_hit,
       AT.Home_runs_per_hit as Away_Team_Home_runs_per_hit
FROM ht_batting_data HT
JOIN at_batting_data AT
ON HT.game_id = AT.game_id
ORDER BY HT.game_id;

DROP TABLE IF EXISTS Final_joint;
CREATE TABLE Final_joint
SELECT ha.game_id,
       ha.local_date,
       ha.home_team,
       ha.away_team,
       ha.home_PA,
       ha.home_BA,
       ha.home_H,
       ha.home_HR,
       ha.home_HR9,
       ha.home_BB,
       ha.home_BB9,
       ha.home_K,
       ha.home_K9,
       ha.home_KBB,
       ha.home_TP,
       ha.home_Flyout,
       ha.home_GIDP,
       ha.home_runs,
       ha.home_errors,
       ha.away_PA,
       ha.away_BA,
       ha.away_H,
       ha.away_HR,
       ha.away_HR9,
       ha.away_BB,
       ha.away_BB9,
       ha.away_K,
       ha.away_K9,
       ha.away_KBB,
       ha.away_TP,
       ha.away_Flyout,
       ha.away_GIDP,
       ha.away_runs,
       ha.away_errors,
       hb.Home_Team_Slugging_Percentage,
       hb.Away_Team_Slugging_Percentage,
       hb.Home_Team_Batting_Average,
       hb.Away_Team_Batting_Average,
       hb.Home_Team_Walk_strikeout_ratio,
       hb.Away_Team_Walk_strikeout_ratio,
       hb.Home_Team_Ground_fly_ball_ratio,
       hb.away_Team_Ground_fly_ball_ratio,
       hb.Home_Team_Intentional_Walk,
       hb.Away_Team_Intentional_Walk,
       hb.Home_Team_At_bats_per_home_run,
       hb.Away_Team_At_bats_per_home_run,
       hb.Home_Team_Home_runs_per_hit,
       hb.Away_Team_Home_runs_per_hit
FROM ha_pitching_joint ha
JOIN ha_batting_joint hb
ON ha.game_id = hb.game_id
GROUP BY ha.game_id
ORDER BY ha.game_id;
CREATE INDEX final_id_index ON Final_joint (game_id);
CREATE INDEX final_d_index ON Final_joint (local_date);
CREATE INDEX final_h_index ON Final_joint (home_team);
CREATE INDEX final_a_index ON Final_joint (away_team);

DROP TABLE IF EXISTS rolling_final_joint;
CREATE TABLE rolling_final_joint
SELECT f.game_id,
       f.local_date,
       f.home_team,
       f.away_team,
       i.HomeTeamWins,
       COUNT(h2.game_id) AS num_games,
       SUM(h2.home_PA) / COUNT(h2.game_id) AS r_home_PA,
       SUM(h2.home_BA) / COUNT(h2.game_id) AS r_home_BA,
       SUM(h2.home_H) / COUNT(h2.game_id) AS r_home_H,
       SUM(h2.home_HR) / COUNT(h2.game_id) AS r_home_HR,
       SUM(h2.home_HR9) / COUNT(h2.game_id) AS r_home_HR9,
       SUM(h2.home_BB) / COUNT(h2.game_id) AS r_home_BB,
       SUM(h2.home_BB9) / COUNT(h2.game_id) AS r_home_BB9,
       SUM(h2.home_K) / COUNT(h2.game_id) AS r_home_K,
       SUM(h2.home_K9) / COUNT(h2.game_id) AS r_home_K9,
       SUM(h2.home_KBB) / COUNT(h2.game_id) AS r_home_KBB,
       SUM(h2.home_TP) / COUNT(h2.game_id) AS r_home_TP,
       SUM(h2.home_Flyout) / COUNT(h2.game_id) AS r_home_Flyout,
       SUM(h2.home_GIDP) / COUNT(h2.game_id) AS r_home_GIDP,
       SUM(h2.home_runs) / COUNT(h2.game_id) AS r_home_runs,
       SUM(h2.home_errors) / COUNT(h2.game_id) AS r_home_errors,
       SUM(h2.away_PA) / COUNT(h2.game_id) AS r_away_PA,
       SUM(h2.away_BA) / COUNT(h2.game_id) AS r_away_BA,
       SUM(h2.away_H) / COUNT(h2.game_id) AS r_away_H,
       SUM(h2.away_HR) / COUNT(h2.game_id) AS r_away_HR,
       SUM(h2.away_HR9) / COUNT(h2.game_id) AS r_away_HR9,
       SUM(h2.away_BB) / COUNT(h2.game_id) AS r_away_BB,
       SUM(h2.away_BB9) / COUNT(h2.game_id) AS r_away_BB9,
       SUM(h2.away_K) / COUNT(h2.game_id) AS r_away_K,
       SUM(h2.away_K9) / COUNT(h2.game_id) AS r_away_K9,
       SUM(h2.away_KBB) / COUNT(h2.game_id) AS r_away_KBB,
       SUM(h2.away_TP) / COUNT(h2.game_id) AS r_away_TP,
       SUM(h2.away_Flyout) / COUNT(h2.game_id) AS r_away_Flyout,
       SUM(h2.away_GIDP) / COUNT(h2.game_id) AS r_away_GIDP,
       SUM(h2.away_runs) / COUNT(h2.game_id) AS r_away_runs,
       SUM(h2.away_errors) / COUNT(h2.game_id) AS r_away_errors,
       SUM(h2.Home_Team_Slugging_Percentage)/COUNT(h2.game_id) AS r_Home_Team_Slugging_Percentage,
       SUM(h2.Away_Team_Slugging_Percentage)/COUNT(h2.game_id) AS r_Away_Team_Slugging_Percentage,
       SUM(h2.Home_Team_Batting_Average)/COUNT(h2.game_id) AS r_Home_Team_Batting_Average,
       SUM(h2.Away_Team_Batting_Average)/COUNT(h2.game_id) AS r_Away_Team_Batting_Average,
       SUM(h2.Home_Team_Walk_strikeout_ratio)/COUNT(h2.game_id) AS r_Home_Team_Walk_strikeout_ratio,
       SUM(h2.Away_Team_Walk_strikeout_ratio)/COUNT(h2.game_id) AS r_Away_Team_Walk_strikeout_ratio,
       SUM(h2.Home_Team_Ground_fly_ball_ratio)/COUNT(h2.game_id) AS r_Home_Team_Ground_fly_ball_ratio,
       SUM(h2.away_Team_Ground_fly_ball_ratio)/COUNT(h2.game_id) AS r_away_Team_Ground_fly_ball_ratio,
       SUM(h2.Home_Team_Intentional_Walk)/COUNT(h2.game_id) AS r_Home_Team_Intentional_Walk,
       SUM(h2.Away_Team_Intentional_Walk)/COUNT(h2.game_id) AS r_Away_Team_Intentional_Walk,
       SUM(h2.Home_Team_At_bats_per_home_run)/COUNT(h2.game_id) AS r_Home_Team_At_bats_per_home_run,
       SUM(h2.Away_Team_At_bats_per_home_run)/COUNT(h2.game_id) AS r_Away_Team_At_bats_per_home_run,
       SUM(h2.Home_Team_Home_runs_per_hit)/COUNT(h2.game_id) AS r_Home_Team_Home_runs_per_hit,
       SUM(h2.Away_Team_Home_runs_per_hit)/COUNT(h2.game_id) AS r_Away_Team_Home_runs_per_hit
FROM Final_joint f
JOIN Final_joint h2
ON f.home_team = h2.home_team
AND h2.local_date
    BETWEEN DATE_SUB(f.local_date, INTERVAL 100 DAY)
    AND DATE_SUB(f.local_date, INTERVAL 1 DAY)
JOIN game_info i
ON f.game_id = i.game_id
GROUP BY f.game_id
ORDER BY f.game_id ASC;


