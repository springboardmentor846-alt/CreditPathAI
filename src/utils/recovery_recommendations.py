def recovery_action(risk_category):
    """
    Map risk levels to recommended recovery interventions.
    """
    if risk_category == "Low Risk":
        return {
            "action": "Automated Reminder",
            "message": "Send SMS/email reminder. Borrower has high repayment probability.",
            "priority": "Low"
        }

    elif risk_category == "Medium Risk":
        return {
            "action": "Agent Follow-up",
            "message": "Schedule a call with borrower to discuss repayment options.",
            "priority": "Medium"
        }

    elif risk_category == "High Risk":
        return {
            "action": "Intensive Recovery Strategy",
            "message": "High chance of default. Consider restructuring, field visit, or escalation.",
            "priority": "High"
        }

    else:
        return {
            "action": "Unknown",
            "message": "Risk category not recognized.",
            "priority": "N/A"
        }
