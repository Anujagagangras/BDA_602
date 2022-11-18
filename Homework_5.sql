SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;
#create table for home team
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

#create table for away team
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

#joining home team table and away team table where home team game id = away team game id
DROP TABLE IF EXISTS filtered_team_pitching_count;
CREATE TABLE filtered_team_pitching_count
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

# create result table
DROP TABLE IF EXISTS Result_table;
CREATE TABLE Result_table
(
SELECT
    g.game_id
    , g.home_team_id
    , home_runs
    , away_runs
    , home_errors
    , away_errors
    , home_hits
    , g.away_team_id
    , away_hits
    , home_atBat
    , away_atBat
    , away_home_run
    , home_home_run
    , away_final_score
    , home_final_score
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
    JOIN filtered_team_pitching_count ft on g.game_id = ft.game_id
    ORDER BY g.game_id
    );

#adding some new features IP,K/9, BB/9,ERA, WHIP
DROP TABLE IF EXISTS Starting_Pitcher_Stats;
CREATE TABLE Starting_Pitcher_Stats(
SELECT
    p.game_id
    , p.team_id
    , p.pitcher
    #Walks and hits per inning pitched
    ,(p.Walk + p.Hit) / (p.endingInning - p.startingInning + 1) AS WHIP
    ,((p.endingInning - p.startingInning + 1)/3) AS Inning_Pitched
    # Strikeouts per 9 innings pitched: strikeouts times nine divided by innings pitched
    ,((p.Strikeout/(p.endingInning - p.startingInning + 1)) * 9) AS K9
    , ((((p.atBat - p.outsPlayed)/2)/(p.endingInning - p.startingInning + 1)) * 9) AS EarnedRunAverage
    # Bases on balls per 9 innings pitched: base on balls multiplied by nine, divided by innings pitched
    , ((p.Walk/(p.endingInning - p.startingInning + 1)/3) * 9) AS BB9

    FROM pitcher_counts p
    JOIN team t ON p.team_id = t.team_id
    ORDER BY p.game_id, p.pitcher
);


DROP TABLE IF EXISTS stats_league_pitching;
CREATE TABLE stats_league_pitching
(
SELECT
    ((SUM((atBat - outsPlayed)/2)/SUM(endingInning - startingInning + 1)) * 9) AS lgEarnedRunAverage
    , SUM(Home_Run) AS lgHomeRun
    , SUM(endingInning - startingInning + 1) AS lgInningPitched
    , AVG(COALESCE((Home_Run/NULLIF((Fly_Out + Flyout),0)),0)) AS lgHomeFlyout
    , SUM(Walk) AS lgBaseOnBall
    , SUM(Strikeout) AS lgStrikeout
    FROM pitcher_counts p
    JOIN team t ON p.team_id = t.team_id
    GROUP BY t.league
);