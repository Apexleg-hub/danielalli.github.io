-- Basic SQL Statements You Need Every Day

--Here are the most common and essential SQL commands, explained with simple examples.

--1. Database Level

CREATE DATABASE company_db;        -- Create a new database
USE company_db;                    -- Select database to work with
DROP DATABASE company_db;          -- Delete entire database (be careful!)


--#### 2. Table Creation & Deletion

-- Create a table
CREATE TABLE employees (
    id INT PRIMARY  KEY,
    name VARCHAR(100) NOT NULL,
    salary DECIMAL(10,2),
    dept VARCHAR(50)
);
 
select * from employees

-- Delete table
DROP TABLE employees;

-- Delete only data (keep table structure)
TRUNCATE TABLE employees;


--3. Insert Data (Add rows)

INSERT INTO employees (id,name, salary, dept) 
VALUES (1,'John Doe', 50000.00, 'IT');

-- Insert multiple rows
INSERT INTO employees (id,name, salary, dept) VALUES
(2,'Alice', 60000, 'HR'),
(3,'Bob', 55000, 'Sales'),
(4,'Eve', 70000, 'IT');

select * from employees

-- 4. SELECT – Read Data

SELECT * FROM employees;                    -- All columns, all rows
SELECT name, salary, dept FROM employees;         -- Only specific columns
SELECT * FROM employees 
WHERE dept = 'IT'; -- Filter with condition

SELECT name, salary, dept FROM employees 
WHERE dept = 'HR'; -- Filter with condition

SELECT * FROM employees
WHERE salary > 60000;

SELECT * FROM employees
WHERE salary = 60000;

SELECT * FROM employees
WHERE salary < 60000;

SELECT * FROM employees
WHERE salary >= 60000;
SELECT * FROM employees
WHERE salary <= 60000;

SELECT name, salary FROM employees
WHERE salary <= 60000;

SELECT * FROM employees ORDER BY salary DESC;   -- Sort (DESC = high to low)
SELECT * FROM employees ORDER BY salary asc;
SELECT * FROM employees ORDER BY dept asc;

SELECT * FROM employees LIMIT  3;             -- Only first 3 rows

SELECT name, dept FROM employees LIMIT  3; 

SELECT name, dept FROM employees
WHERE salary <= 60000
LIMIT  3; 

SELECT * FROM employees;
-- 5. UPDATE – Modify Existing Data

UPDATE employees 
SET salary = 65000 
WHERE name = 'Alice';

-- Increase everyone in IT by 10%
UPDATE employees 
SET salary = salary * 10 
WHERE dept = 'IT';


--6. DELETE – Remove Rows

DELETE FROM employees WHERE id = 5;           -- Delete specific row
DELETE FROM employees WHERE dept = 'HR';      -- Delete all in HR
DELETE FROM employees;                        -- Delete ALL rows (table becomes empty)


-- 7. Common WHERE Conditions

WHERE salary >= 50000 AND dept = 'IT'
WHERE name LIKE '%ohn%'          -- Contains "ohn"
WHERE name LIKE 'A%'             -- Starts with A
WHERE salary BETWEEN 50000 AND 80000
WHERE dept IN ('IT', 'Sales', 'HR')
WHERE salary IS NULL             -- or IS NOT NULL


--8. Aggregation (Counting, Summing, etc.)

SELECT COUNT(*) FROM employees;                    -- Total rows
SELECT AVG(salary) AS avg_salary FROM employees;  -- Average salary
SELECT MAX(salary) AS highest FROM employees;
SELECT MIN(salary) AS lowest FROM employees;
SELECT dept, AVG(salary) FROM employees GROUP BY dept;
SELECT dept, COUNT(*) FROM employees GROUP BY dept HAVING COUNT(*) > 2;


--9. JOIN Tables (Most Important!)

-- Sample second table
CREATE TABLE departments (
    dept_id INT PRIMARY KEY,
    dept_name VARCHAR(50)
);

-- Inner Join (most common)
SELECT e.name, e.salary, d.dept_name
FROM employees e
JOIN departments d ON e.dept = d.dept_name;

-- Left Join (all employees, even if dept not in departments table)
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept = d.dept_name;

/*Top 10 Commands You’ll Use 95% of the Time
| Command     | Purpose                          |
|-------------|----------------------------------|
| SELECT      | Get data                         |
| INSERT      | Add new rows                     |
| UPDATE      | Change existing data             |
| DELETE      | Remove rows                      |
| WHERE       | Filter results                   |
| ORDER BY    | Sort results                     |
| LIMIT       | Get only first N rows            |
| JOIN        | Combine multiple tables          |
| GROUP BY    | Summarize (with COUNT, AVG, etc.)|
| CREATE TABLE| Make a new table                 |*/

--Quick Cheat Sheet (Copy-Paste)

SELECT * FROM table_name;
SELECT column1, column2 FROM table_name WHERE condition ORDER BY column LIMIT 10;

INSERT INTO table_name (col1, col2) VALUES ('val1', 123);

UPDATE table_name SET column = 'new value' WHERE id = 5;

DELETE FROM table_name WHERE id = 10;


