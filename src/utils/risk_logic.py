def risk_category(prob):
    if prob < 0.4:
        return "Low Risk"
    elif prob < 0.7:
        return "Medium Risk"
    else:
         "High Risk"
