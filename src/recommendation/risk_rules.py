def assign_risk(prob):
    if prob < 0.30:
        return "Low Risk", "No action required"
    elif prob < 0.60:
        return "Medium Risk", "Send payment reminder & financial counseling"
    else:
        return "High Risk", "Immediate recovery action / loan restructuring"
