SELECT /*+ PARALLEL(4) */ 
  f.fleet_id,
  f.company_no,
  TRUNC(end_date) AS day_id,
  COUNT(DISTINCT CASE WHEN r.is_drive = 1 THEN r.booking_id END) AS cnt_drives,
  COUNT(DISTINCT vehicle_no) AS cnt_veh_used,
  SUM(income_netto) AS sum_income_netto,
  SUM(kilometers) AS sum_kilometers,
  SUM(minutes) AS sum_minutes
FROM dn_fact_prd_usage_details_v2 r
INNER JOIN dn_dim_fleet_view f ON r.return_city = f.city_id
INNER JOIN dn_dim_customer_view cst ON r.customer_id = cst.customer_id
WHERE 1 = 1
  AND f.fleet_id NOT IN ('SFO', 'STH', 'BB1')
  AND cst.customer_group IN ('CUSTOMER')
  AND r.booking_date >= TO_DATE('01.03.2018', 'dd.mm.yyyy')
  AND r.end_date BETWEEN TO_DATE('01.03.2018', 'dd.mm.yyyy') AND TO_DATE('15.10.2018', 'dd.mm.yyyy') - INTERVAL '1' SECOND
GROUP BY  
  f.fleet_id,
  f.company_no,
  TRUNC(end_date)
ORDER BY 1, 2, 3  
;  
