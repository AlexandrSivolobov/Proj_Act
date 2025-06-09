SELECT 
    p.ProductID,
    p.Name AS ProductName,
    SUM(sod.OrderQty) AS TotalSold,
    SUM(sod.OrderQty * sod.UnitPrice * (1 - sod.UnitPriceDiscount)) AS TotalRevenue
FROM 
    sales.salesorderdetail sod
    INNER JOIN production.product p ON sod.ProductID = p.ProductID
GROUP BY 
    p.ProductID, p.Name
HAVING 
    SUM(sod.OrderQty) > 50
ORDER BY 
    TotalSold DESC;