CREATE TABLE customers (
    customer_id INT PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    country VARCHAR(50),
    join_date DATE
);

CREATE TABLE products (
    product_id INT PRIMARY KEY,
    category VARCHAR(50),
    sub_category VARCHAR(50),
    product_name VARCHAR(100)
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date DATE,
    payment_method VARCHAR(50),
    marketing_channel VARCHAR(50),
    total_amount NUMERIC
);

CREATE TABLE order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT,
    unit_price NUMERIC
);

-- Load data

-- query

--1.Total Revenue and Orders per Month
SELECT
    DATE_TRUNC('month', order_date) AS month,
    COUNT(order_id) AS total_orders,
    SUM(total_amount) AS total_revenue
FROM orders
GROUP BY DATE_TRUNC('month', order_date)
ORDER BY month;

--2. Month-Over-Month Revenue Growth
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(total_amount) AS revenue
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date)
)
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month,
    ROUND(
        (revenue - LAG(revenue) OVER (ORDER BY month))
        / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) * 100,
        2
    ) AS mom_growth_pct
FROM monthly
ORDER BY month;

-- 3. Best-Selling Products and Revenue Ranking
SELECT
    p.product_id,
    p.product_name,
    SUM(oi.quantity) AS total_units_sold,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    RANK() OVER (ORDER BY SUM(oi.quantity * oi.unit_price) DESC) AS revenue_rank
FROM order_items oi
JOIN products p USING (product_id)
GROUP BY p.product_id, p.product_name
ORDER BY total_revenue DESC;

--4. Customer Lifetime Value (LTV)
SELECT
    customer_id,
    COUNT(order_id) AS total_orders,
    SUM(total_amount) AS lifetime_value,
    AVG(total_amount) AS avg_order_value
FROM orders
GROUP BY customer_id
ORDER BY lifetime_value DESC;

--5. Cohort Retention Analysis
WITH first_order AS (
    SELECT customer_id, MIN(order_date) AS cohort_month
    FROM orders
    GROUP BY customer_id
),
activity AS (
    SELECT
        o.customer_id,
        DATE_TRUNC('month', fo.cohort_month) AS cohort,
        DATE_TRUNC('month', o.order_date) AS active_month
    FROM orders o
    JOIN first_order fo USING (customer_id)
)
SELECT
    cohort,
    active_month,
    COUNT(DISTINCT customer_id) AS active_users
FROM activity
GROUP BY cohort, active_month
ORDER BY cohort, active_month;

--6. Marketing Channel Performance
SELECT
    marketing_channel,
    COUNT(order_id) AS orders_count,
    SUM(total_amount) AS revenue,
    ROUND(SUM(total_amount) / COUNT(order_id), 2) AS avg_order_value
FROM orders
GROUP BY marketing_channel
ORDER BY revenue DESC;

--7. Low-Performing Products 
SELECT
    p.product_id,
    p.product_name,
    SUM(oi.quantity) AS units_sold
FROM products p
LEFT JOIN order_items oi USING (product_id)
GROUP BY p.product_id, p.product_name
HAVING SUM(oi.quantity) < 20
ORDER BY units_sold ASC;

--8. Rolling 7-Day Average Sales (Demand Forecast Support)
SELECT
    order_date,
    SUM(total_amount) AS daily_sales,
    AVG(SUM(total_amount)) OVER (
        ORDER BY order_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7_day_avg_sales
FROM orders
GROUP BY order_date
ORDER BY order_date;