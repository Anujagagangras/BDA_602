SHOW ERRORS;
SHOW WARNINGS;

USE baseball;
SHOW TABLES;

## to test annual average
select * from Annual_Average
         where batter=110029;

## to test historic average
select * from Historic_Average
         where batter=110029;

## to test rolling average
select * from Rolling_Average
         where batter=110029 limit 0,30;