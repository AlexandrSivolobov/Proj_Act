SELECT 
    soh.SalesOrderID,
    soh.OrderDate,
    soh.TotalDue,
    soh.CustomerID
FROM 
    sales.salesorderheader soh
WHERE 
    soh.TotalDue > 10000
    AND soh.OrderDate BETWEEN '2012-01-01' AND '2012-12-31'
ORDER BY 
    soh.OrderDate DESC, soh.TotalDue DESC;