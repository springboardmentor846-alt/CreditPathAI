def risk_category(prob):
    if prob >= 0.75:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"