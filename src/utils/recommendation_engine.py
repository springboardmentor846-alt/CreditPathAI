def recommend_action(risk_category):
    """
    Map a borrower's risk category to a recommended recovery strategy.
    """

    if risk_category == "Low Risk":
        return {
            "action": "Send gentle reminder",
            "message": "Borrower is low risk. Send a friendly payment reminder via email/SMS.",
            "priority": "Low"
        }

    elif risk_category == "Medium Risk":
        return {
            "action": "Call the borrower",
            "message": "Moderate risk. Suggest calling borrower to discuss payment options.",
            "priority": "Medium"
        }

    elif risk_category == "High Risk":
        return {
            "action": "Escalate to recovery team",
            "message": "High risk of default. Assign to senior collections specialist immediately.",
            "priority": "High"
        }

    else:
        return {
            "action": "Unknown",
            "message": "Risk category not recognized.",
            "priority": "None"
        }
