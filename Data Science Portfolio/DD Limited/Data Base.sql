create table trial_balance(
CODE int,
DESCRIPTION text,
DR_CR int,
FINANCIAL_STATEMENT_LINE varchar (150)
);

create table trialbalance(
CODE int,
DESCRIPTION text,
DR_CR int,
FINANCIAL_STATEMENT_LINE varchar (150)
);

select * from trial_balance
insert into trial_balance (CODE,
DESCRIPTION,
DR_CR,
FINANCIAL_STATEMENT_LINE )
Values(5000-001,	'Paid Up Capital',	-2000000000.00,	'Equity & Reserves'),
(5290-001,	'Cumulative Profit & Loss A/c',   -1659538640.00,	'Equity & Reserves');