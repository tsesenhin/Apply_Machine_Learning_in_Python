      
--Question1 Part 1 PostgreSQL 9.6

select count(*) from faceit
where created_at >= '2018-01-01' and created_at <='2018-12-31'
and membership = 'premium'


--Question1 Part 2 PostgreSQL 9.6

select 
distinct user_id 

from

(select user_id, match_id, game, created_at, membership, faction, winner,
rank()Over(PARTITION BY user_id,game order by user_id,created_at asc) as rank1 
from FACEIT
where faction = winner
group by user_id, match_id, game, created_at, membership, faction, winner) as table1

where rank1 >= 3





select 

q.query_name,

round(avg((cast(rating as float) /cast(position as float))),2) AS quality
,

round(cast(sum(case when q.rating < 3 then 1 else 0 end) as float)/cast(count(q.rating) as float)*100,2)
as poor_query_percentage

from queries AS q 

group by q.query_name


