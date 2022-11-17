SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;

DROP TABLE IF EXISTS home_pitching_counts;
CREATE TABLE home_pitching_counts
    (   select tpc.game_id
           , tpc.team_id as home_team_id
           , finalScore as home_final_score
           , atBat as home_atBat
           , Hit as home_Hit
           , Home_Run as home_home_run
        from team_pitching_counts tpc
        group by game_id, homeTeam
        having homeTeam = 1);

DROP TABLE IF EXISTS away_pitching_counts;
CREATE TABLE away_pitching_counts
    (   select tpc.game_id
           , tpc.team_id as away_team_id
           , finalScore as away_final_score
           , atBat as away_atBat
           , Hit as away_Hit
           , Home_Run as away_home_run
        from team_pitching_counts tpc
        group by game_id, awayTeam
        having awayTeam = 1);

DROP TABLE IF EXISTS ah_table;
CREATE TABLE ah_table
(
    select apc.game_id
    ,away_atBat
    ,home_atBat
    ,away_home_run
    ,home_home_run
    ,away_final_score
    ,home_final_score
    from away_pitching_counts apc
    join home_pitching_counts hpc on apc.game_id = hpc.game_id
);


SELECT
    g.game_id
    , g.home_team_id
    , home_runs
    , home_errors
    , home_hits
    , home_atBat
    , away_atBat
    , CASE
        WHEN home_runs > away_runs  THEN  '1'
        WHEN home_runs < away_runs THEN '0'
        ELSE  '2' END AS Home_Team_Wins
    ,away_t.name AS away_team
    ,home_t.name AS home_team
    FROM boxscore bx
    JOIN game g ON g.game_id = bx.game_id
    JOIN team away_t ON g.away_team_id = away_t.team_id
    JOIN team home_t ON g.home_team_id = home_t.team_id
    JOIN ah_table aht on g.game_id = aht.game_id
    ORDER BY g.game_id;

#################################################################################################
CREATE TABLE IF NOT EXISTS league_stats(
    SELECT
        t.league
        , ((SUM((atBat - outsPlayed)/2)/SUM(endingInning - startingInning + 1)) * 9) AS lgERA
        , SUM(Home_Run) AS lgHR
        , SUM(Walk) AS lgBB
        , SUM(Strikeout) AS lgK
        , SUM(endingInning - startingInning + 1) AS lgIP
        , AVG(COALESCE((Home_Run/NULLIF((Fly_Out + Flyout),0)),0)) AS lgHRFB
    FROM pitcher_counts p
        JOIN team t ON p.team_id = t.team_id
    GROUP BY t.league
);

