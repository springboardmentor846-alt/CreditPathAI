from src.recommendation.recommend import recommend_action

sample_customer = {
    "Age": 35,
    "Income": 60000,
    "LoanAmount": 15000,
    "CreditScore": 650,
    "MonthsEmployed": 48,
    "NumCreditLines": 4,
    "InterestRate": 12.5,
    "LoanTerm": 36,
    "DTIRatio": 0.35,
    "LoanIncomeRatio": 0.25
}

print(recommend_action(sample_customer))
