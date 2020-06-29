DROP TABLE IF EXISTS xxx;

SET hive.auto.convert.join=False;


CREATE TABLE xxx
STORED AS PARQUET
LOCATION '/xxxx/xxxx/xxxx'


as


With ft2 as (SELECT iaimsslno,vchaircraftregistrationid, vchfltnumber as vchfltnumber2, dtdateofflight as dtdateofflight2,vchactualdepairportcode,
vchactualarrairportcode,dtstdutc, dtstautc, dtetdutc, dtetautc,iseatssold,City2.city, City2.airportcode, City2.airportname2
, tirecordstatus,adh_statusflag,iaimsstatus,ieresslno  FROM raw_flighttracker_delta.eresaims_flightstatus

INNER JOIN (SELECT city, airportcode,airportname as airportname2 FROM raw_eresrevenue_delta.airportcode
            WHERE inuseflag LIKE 'X'
            AND tirecordstatus = X
            AND adh_statusflag LIKE 'U' OR adh_statusflag LIKE 'X'
            ) City2
            ON vchactualarrairportcode = City2.airportcode

WHERE tirecordstatus = X
AND (vchaircraftregistrationid NOT LIKE '%X%' OR  vchaircraftregistrationid NOT LIKE '%X%')
AND (adh_statusflag LIKE 'X' OR adh_statusflag LIKE 'X')
AND iaimsstatus = X
AND ieresslno IS NOT NULL

)


SELECT 

ft1.vchfltnumber as ft1_vchfltnumber,
to_date(ft1.dtdateofflight) as ft1_dtdateofflight,
ft1.vchactualdepairportcode as ft1_vchactualdepairportcode, 
ft1.vchactualarrairportcode as ft1_vchactualarrairportcode,
ft1.dtstdutc as ft1_dtstdutc,
ft1.dtstautc as ft1_dtstautc,
ft1.dtetdutc as ft1_dtetdutc,
ft1.dtetautc as ft1_dtetautc,
ft1.dtstautc as ft1_stautc,
ft2.dtstdutc as ft2_stdutc,
ft1.adh_updatedate as ft1_adh_updatedate,


                
Concat(SUBSTRING(cast(ft1.dtetdutc as string),12,8), ' - ', 
       SUBSTRING(cast(CASE
                      WHEN ft1.dtetautc IS NOT NULL THEN ft1.dtetautc
                      ELSE ft1.dtstautc
                      END as string),12,8)) as ETD_to_STA_ETA_string,
                      
Concat(SUBSTRING(cast(ft1.dtetdutc as string),12,8), ' - ', 
       SUBSTRING(cast(CASE
                      WHEN ft2.dtetautc IS NOT NULL THEN ft2.dtetautc
                      ELSE ft2.dtstautc
                      END as string),12,8)) as ETD_to_STA_ETA_string_indirect,    
   
   datediff(ft2.dtdateofflight2,ft1.dtdateofflight) as check,
    
     
     
Case
     WHEN datediff(ft2.dtdateofflight2,ft1.dtdateofflight) = 0 THEN 'X'
     WHEN datediff(ft2.dtdateofflight2,ft1.dtdateofflight) = 1 THEN 'X'
     Else 'X'
     END as Same_Nextday_flight

,ft1.iseatssold as ft1_iseatssold,
Aircap.iaircraftcapacity,
(Aircap.iaircraftcapacity- ft1.iseatssold) as ft1_availableseats,
ft2.vchfltnumber2 as ft2_vchfltnumber,
to_date(ft2.dtdateofflight2) as ft2_dtdateofflight,
ft2.vchactualdepairportcode as ft2_vchactualdepairportcode, 
ft2.vchactualarrairportcode as ft2_vchactualarrairportcode, 
ft2.dtstdutc as ft2_dtstdutc,
ft2.dtstautc as ft2_dtstautc, 
ft2.dtetdutc as ft2_dtetdutc,
ft2.dtetautc as ft2_dtetautc,
Aircap2.iaircraftcapacity2 as Aircap2_iaircraftcapacity,
ft2.iseatssold as ft2_iseatssold,
(Aircap2.iaircraftcapacity2- ft2.iseatssold) as ft2_availableseats, 



CASE
     WHEN ft1.dtetautc IS NOT NULL THEN ft1.dtetautc
     ELSE ft1.dtstautc
     END flt1_soonestarrival,
     

CASE
     WHEN ft2.dtetautc IS NOT NULL THEN ft2.dtetautc
     ELSE ft2.dtstautc
     END flt1n2_soonestarrival,
     


Concat(ft1.vchactualdepairportcode, ' - ' ,ft1.vchactualarrairportcode) AS Ft1_Route,

Concat(ft1.vchactualdepairportcode, ' - ' ,ft1.vchactualarrairportcode, ' - ', ft2.vchactualarrairportcode) AS Indirect_Route,

Concat(ft1.vchfltnumber, ' - ' ,ft2.vchfltnumber2) AS Ft1n2fltnumber,

Concat(cast((Aircap.iaircraftcapacity- ft1.iseatssold)as string), '/',cast((Aircap2.iaircraftcapacity2- ft2.iseatssold)as string)) AS ft1_n_ft2_Available_seats_string,

CONCAT(

     case when length((cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string))) = X THEN CONCAT('0', (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string)))
     ELSE (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string))
END,

':',

case when length((cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string))) = X THEN CONCAT('0', (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string)))
ELSE (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string))
END) AS Time_to_Departure,

concat(
Case 
     WHEN length(cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)/60 as int) as string)) = X THEN concat('0', cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)/60 as int) as string))
     ELSE cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)/60 as int) as string)
     END,

     ':',

     Case 
     WHEN length(cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)%60 as int) as string)) = X THEN concat('0', cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)%60 as int) as string))
     ELSE cast(cast(((unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60)%60 as int) as string)
     END

      )Waiting_Time_HHMM,



(unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60 AS Time_to_departure_in_minutes,


Concat(ft1.vchactualdepairportcode, ' - ' ,ft1.vchactualarrairportcode, ' - ', ft2.vchactualarrairportcode) As Direct_flight_square,

Concat(ft1.vchfltnumber, ' - ',ft2.vchfltnumber2) AS Direct_flight_number_square,

Concat(SUBSTRING(cast(ft1.dtetdutc as string),12,5), ' - ', 
       SUBSTRING(cast(CASE
                      WHEN ft2.dtetautc IS NOT NULL THEN ft2.dtetautc
                      ELSE ft2.dtstautc
                      END as string),12,5)) as Direct_ETDtoSTA_ETA_square,
                      
Concat(cast((Aircap.iaircraftcapacity- ft1.iseatssold)as string), '/',cast((Aircap2.iaircraftcapacity2- ft2.iseatssold)as string)) AS Direct_Availableseats_square,



CONCAT(ft2.vchactualarrairportcode, ' - ', (CASE
    WHEN ft2.airportname2 = 'X' OR ft2.airportname2 = 'x'  THEN 'North West England'
    
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x'  THEN 'Scotland'
    
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Crete'
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Sicily' 
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Sardinia'
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Corsica'
    WHEN ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' OR ft2.airportname2 ='x'
         THEN 'Canary Islands'  
         
    WHEN  ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Swiss Sector'
    WHEN  ft2.airportname2 = 'x' OR ft2.airportname2 = 'x' THEN 'Andalucia'
    WHEN  ft2.airportname2 = 'x' OR airportname2 = 'x' THEN 'Milano'     
         
    WHEN ft2.city = 'x' THEN 'x'
    WHEN ft2.city = 'x' THEN 'x'
    ELSE ft2.city     
    END)) ft1_arrival_city_region,


ft2.airportcode as ft2_arr_airportcode,


CONCAT(ft1.vchactualdepairportcode, ' - ',(
CASE
    WHEN City3.airportname = 'x' OR  City3.airportname = 'x' THEN 'North West England'
    
    WHEN City3.airportname = 'x' OR   City3.airportname = 'x'  THEN 'Scotland'
    
    WHEN City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Crete'
    WHEN City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Sicily' 
    WHEN City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Sardinia'
    WHEN City3.airportname = 'x' OR City3.airportname = 'x' OR City3.airportname = 'Figari' THEN 'Corsica'
    WHEN City3.airportname = 'x' OR City3.airportname = 'x' OR City3.airportname = 'Gran Canaria' OR City3.airportname = 'La Palma' OR City3.airportname ='Tenerife'
         THEN 'Canary Islands' 
         
    WHEN  City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Swiss Sector'
    WHEN  City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Andalucia'
    WHEN  City3.airportname = 'x' OR City3.airportname = 'x' THEN 'Milano'          
         
    WHEN City3.city = 'x' THEN 'Bodrum'
    WHEN City3.city = 'x' THEN 'Isle of Man'     
    ELSE City3.city     
    END)) ft1_dep_airport_city_region,


City3.airportcode as ft1_dep_airportcode,

CONCAT(ft2.vchactualdepairportcode, ' - ', ft2.vchactualarrairportcode) AS f1_route,

CONCAT('(STD) ', SUBSTRING(cast(ft1.dtstdutc as string),12,5), ' - ',  SUBSTRING(cast(ft2.dtstautc as string),12,5),' (STA)') AS ft1_STD_STA,


CONCAT('(ETD) ',SUBSTRING(cast(ft1.dtetdutc as string),12,5), ' - ',  CASE 
                                                             WHEN SUBSTRING(cast(ft2.dtetautc as string),12,5) IS NULL THEN '          '
                                                             ELSE SUBSTRING(cast(ft2.dtetautc as string),12,5)
                                                             END, ' (ETA)') AS ft1_ETDtoETA,
                                                             

Case 
     WHEN (Aircap.iaircraftcapacity- ft1.iseatssold)> (Aircap2.iaircraftcapacity2- ft2.iseatssold) THEN (Aircap2.iaircraftcapacity2- ft2.iseatssold)
     WHEN (Aircap2.iaircraftcapacity2- ft2.iseatssold)> (Aircap.iaircraftcapacity- ft1.iseatssold) THEN (Aircap.iaircraftcapacity- ft1.iseatssold)
     WHEN (Aircap2.iaircraftcapacity2- ft2.iseatssold)= (Aircap.iaircraftcapacity- ft1.iseatssold) THEN (Aircap.iaircraftcapacity- ft1.iseatssold)
     END as Max_Passengers,
     
0 AS total_passengers_direct


FROM raw_flighttracker_delta.eresaims_flightstatus as ft1 



LEFT JOIN  ft2  
           ON ft1.vchactualarrairportcode = ft2.vchactualdepairportcode

           
           
INNER JOIN (SELECT vchaircraftregistration,iaircraftcapacity,tirecordstatus,adh_statusflag
            FROM raw_aims_ods_delta.aircraft
            WHERE tirecordstatus = x
            AND (adh_statusflag LIKE 'x' OR adh_statusflag LIKE 'x')) Aircap
            ON ft1.vchaircraftregistrationid = Aircap.vchaircraftregistration


INNER JOIN (SELECT vchaircraftregistration,iaircraftcapacity as iaircraftcapacity2, tirecordstatus,adh_statusflag
            FROM raw_aims_ods_delta.aircraft
            WHERE tirecordstatus = x
            AND (adh_statusflag LIKE 'x' OR adh_statusflag LIKE 'x')) Aircap2
            ON ft2.vchaircraftregistrationid = Aircap2.vchaircraftregistration
            
            
INNER JOIN (SELECT city, airportcode,airportname FROM raw_eresrevenue_delta.airportcode
            WHERE inuseflag LIKE 'x'
            AND tirecordstatus = x
            AND adh_statusflag LIKE 'x' OR adh_statusflag LIKE 'x') City3
            ON ft1.vchactualdepairportcode = City3.airportcode

           
      

    WHERE ft1.dtstdutc >= current_timestamp()
AND ft1.tirecordstatus = 1
AND (ft1.vchaircraftregistrationid NOT LIKE '%x%' OR  ft1.vchaircraftregistrationid NOT LIKE '%x%')
AND (ft1.adh_statusflag LIKE 'x' OR ft1.adh_statusflag LIKE 'x')
AND ft1.iaimsstatus = x
AND ft1.ieresslno IS NOT NULL
AND (Aircap2.iaircraftcapacity2- ft2.iseatssold) > x AND (Aircap.iaircraftcapacity- ft1.iseatssold) > x
AND (unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60 >= 30

AND ft1.iaimsslno <> ft2.iaimsslno
AND (unix_timestamp(ft2.dtstdutc)-unix_timestamp(ft1.dtstautc))/60 >=120
AND datediff(ft2.dtdateofflight2,ft1.dtdateofflight) BETWEEN 0 AND 1

UNION ALL 




SELECT 

ft1.vchfltnumber as ft1_vchfltnumber,
to_date(ft1.dtdateofflight) as ft1_dtdateofflight,
ft1.vchactualdepairportcode as ft1_vchactualdepairportcode, 
ft1.vchactualarrairportcode as ft1_vchactualarrairportcode,
ft1.dtstdutc as ft1_dtstdutc,
ft1.dtstautc as ft1_dtstautc,
ft1.dtetdutc as ft1_dtetdutc,
ft1.dtetautc as ft1_dtetautc,
ft1.dtstautc as ft1_stautc,
Null as ft2_stdutc,
ft1.adh_updatedate as ft1_adh_updatedate,


                
Concat(SUBSTRING(cast(ft1.dtetdutc as string),12,8), ' - ', 
       SUBSTRING(cast(CASE
                      WHEN ft1.dtetautc IS NOT NULL THEN ft1.dtetautc
                      ELSE ft1.dtstautc
                      END as string),12,8)) as ETD_to_STA_ETA_string,
                      
Null as ETD_to_STA_ETA_string_indirect,    
   
Null as check,
    
     
     
Null as Same_Nextday_flight

,ft1.iseatssold as ft1_iseatssold,
Aircap.iaircraftcapacity,
(Aircap.iaircraftcapacity- ft1.iseatssold) as ft1_availableseats,
Null as ft2_vchfltnumber,
Null as ft2_dtdateofflight,
Null as ft2_vchactualdepairportcode, 
Null as ft2_vchactualarrairportcode, 
Null as ft2_dtstdutc,
Null as ft2_dtstautc, 
Null as ft2_dtetdutc,
Null as ft2_dtetautc,
Null as Aircap2_iaircraftcapacity,
Null as ft2_iseatssold,
Null as ft2_availableseats,


     

CASE
     WHEN ft1.dtetautc IS NOT NULL THEN ft1.dtetautc
     ELSE ft1.dtstautc
     END flt1_soonestarrival,
     

     
Null as flt1n2_soonestarrival,
     
    
     



Concat(ft1.vchactualdepairportcode, ' - ' ,ft1.vchactualarrairportcode) AS Ft1_Route,  

Null AS Indirect_Route,

Null AS Ft1n2fltnumber,

Null AS ft1_n_ft2_Available_seats_string,

CONCAT(

     case when length((cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string))) = x THEN CONCAT('0', (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string)))
     ELSE (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)/60 as int) as string))
END,

':',

case when length((cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string))) = x THEN CONCAT('0', (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string)))
ELSE (cast(cast(((unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60)%60 as int) as string))
END) AS Time_to_Departure,

Null AS Waiting_Time_HHMM,

(unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60 AS Time_to_departure_in_minutes,

Concat(ft1.vchactualdepairportcode, ' - ' ,ft1.vchactualarrairportcode) AS Direct_flight_square,

ft1.vchfltnumber AS Direct_flight_number_square,

Concat(SUBSTRING(cast(ft1.dtetdutc as string),12,5), ' - ', 
       SUBSTRING(cast(CASE
                      WHEN ft1.dtetautc IS NOT NULL THEN ft1.dtetautc
                      ELSE ft1.dtstautc
                      END as string),12,5)) as Direct_ETDtoSTA_ETA_square,
                      
cast((Aircap.iaircraftcapacity- ft1.iseatssold) as string) as Direct_Availableseats_square,

CONCAT(ft1.vchactualarrairportcode, ' - ',
(CASE
    WHEN City5.airportname = 'x' OR  City5.airportname LIKE 'x' THEN 'North West England'
    
    WHEN City5.airportname = 'x' OR City5.airportname LIKE 'x' THEN 'Scotland'
    
    WHEN City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Crete'
    WHEN City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Sicily' 
    WHEN City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Sardinia'
    WHEN City5.airportname = 'x' OR City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Corsica'
    WHEN City5.airportname = 'x' OR City5.airportname = 'x' OR City5.airportname = 'x' OR City5.airportname = 'x' OR City5.airportname ='x'
         THEN 'Canary Islands'   
    
    WHEN  City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Swiss Sector'
    WHEN  City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Andalucia'
    WHEN  City5.airportname = 'x' OR City5.airportname = 'x' THEN 'Milano'
         
     WHEN City5.city = 'x' THEN 'Bodrum'
    WHEN City5.city = 'x' THEN 'Isle of Man'     
    
    
    ELSE City5.city     
    END)) ft1_arrival_city_region,



Null as ft2_arr_airportcode, 

CONCAT( ft1.vchactualdepairportcode, ' - ', 

(CASE
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' THEN 'North West England'
    
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Scotland'
   
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Crete'
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Sicily' 
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Sardinia'
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Corsica'
    WHEN City4.airportname = 'x' OR City4.airportname = 'x' OR City4.airportname = 'x' OR City4.airportname = 'x' OR City4.airportname ='x'
         THEN 'Canary Islands'  
    
    WHEN  City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Swiss Sector'
    WHEN  City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Andalucia'
    WHEN  City4.airportname = 'x' OR City4.airportname = 'x' THEN 'Milano'     
         
     WHEN City4.city = 'x' THEN 'Bodrum'
    WHEN City4.city = 'x' THEN 'Isle of Man'          
    ELSE City4.city     
    END)) ft1_dep_airport_city_region,


Null as ft1_dep_airportcode,--

CONCAT(ft1.vchactualdepairportcode, ' - ', ft1.vchactualarrairportcode) AS f1_route,

CONCAT('(STD) ', SUBSTRING(cast(ft1.dtstdutc as string),12,5), ' - ',  SUBSTRING(cast(ft1.dtstautc as string),12,5),' (STA)') AS ft1_STD_STA,


CONCAT('(ETD) ',SUBSTRING(cast(ft1.dtetdutc as string),12,5), ' - ',  CASE 
                                                             WHEN SUBSTRING(cast(ft1.dtetautc as string),12,5) IS NULL THEN '          '
                                                             ELSE SUBSTRING(cast(ft1.dtetautc as string),12,5)
                                                             END, ' (ETA)') AS ft1_ETDtoETA,
                                                             
0 as Max_Passengers,

(Aircap.iaircraftcapacity- ft1.iseatssold) AS total_passengers_direct



FROM raw_flighttracker_delta.eresaims_flightstatus as ft1 




           
           
           
           
INNER JOIN (SELECT vchaircraftregistration,iaircraftcapacity, tirecordstatus,adh_statusflag
            FROM raw_aims_ods_delta.aircraft
            WHERE tirecordstatus = x
            AND (adh_statusflag = 'x' OR adh_statusflag = 'x')) Aircap
            ON ft1.vchaircraftregistrationid = Aircap.vchaircraftregistration

INNER JOIN (SELECT city, airportcode,airportname FROM raw_eresrevenue_delta.airportcode
            WHERE inuseflag = 'x'
            AND tirecordstatus = x
            AND adh_statusflag = 'x' OR adh_statusflag = 'x') City4
            ON ft1.vchactualdepairportcode = City4.airportcode
            
INNER JOIN (SELECT city, airportcode, airportname FROM raw_eresrevenue_delta.airportcode
            WHERE inuseflag LIKE 'x'
            AND tirecordstatus = x
            AND adh_statusflag = 'x' OR adh_statusflag = 'x') City5
            ON ft1.vchactualarrairportcode = City5.airportcode
            
            
WHERE ft1.dtstdutc >= current_timestamp()
AND ft1.tirecordstatus = x
AND (ft1.vchaircraftregistrationid NOT LIKE 'x' OR  ft1.vchaircraftregistrationid NOT LIKE 'x')
AND (ft1.adh_statusflag LIKE 'x' OR ft1.adh_statusflag LIKE 'x')
AND ft1.iaimsstatus = x
AND ft1.ieresslno IS NOT NULL
AND (Aircap.iaircraftcapacity- ft1.iseatssold) > x
AND (unix_timestamp(ft1.dtetdutc)-unix_timestamp(current_timestamp()))/60 >= xx

