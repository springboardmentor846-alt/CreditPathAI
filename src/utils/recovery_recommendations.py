def recovery_action(category):
    if category == "High Risk":
        return "Immediate personal call & restructuring suggestion."
    elif category == "Medium Risk":
        return "Send reminder SMS and follow-up call."
    else:
        return "Standard repayment reminder."