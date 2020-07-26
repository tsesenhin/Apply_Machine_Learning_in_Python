

--The below can be used for accumlation score for a certain column (post MS SQL SERVER 2012)


select gender, day,   

SUM(score_points) OVER(partition by gender ORDER BY gender, day asc 
     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) 
          AS total
          from scores