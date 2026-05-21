import pandas as pd
import numpy as np

def feature_engineering(data):

    df = data.copy()
    # =========================
    # 1. Debt-to-Income Ratio
    # =========================
    # Financial stress indicator

    df['debt_to_income_ratio'] = (
        df['total_outstanding_debt'] / (df['annual_income'] + 1)
    )

    # =========================
    # 2. Loan-to-Income Ratio
    # =========================
    # Measures affordability of requested loan

    df['loan_to_income_ratio'] = (
        df['loan_application_amount'] / (df['annual_income'] + 1)
    )

    # =========================
    # 3. Debt Per Loan
    # =========================
    # Average debt burden per active loan

    df['debt_per_loan'] = (
        df['total_outstanding_debt'] / (df['num_open_loans'] + 1)
    )

    # =========================
    # 4. Late Payment Ratio
    # =========================
    # Captures repayment behavior

    df['late_payment_ratio'] = (
        df['late_payment_count'] / (df['num_open_loans'] + 1)
    )

    # =========================
    # 5. Spending-to-Balance Ratio
    # =========================
    # Indicates aggressive spending behavior

    df['spending_balance_ratio'] = (
        df['debit_card_spending'] / (df['avg_monthly_balance'] + 1)
    )

    # =========================
    # 6. Digital Activity Score
    # =========================
    # Customer engagement indicator

    df['digital_activity_score'] = (
        df['mobile_banking_logins'] +
        df['online_transfer_frequency']
    )

    # =========================
    # 7. Total Transaction Activity
    # =========================
    # Overall banking activity

    df['total_transaction_activity'] = (
        df['num_deposits_per_month'] +
        df['debit_card_usage_frequency'] +
        df['atm_withdrawal_frequency']
    )

    # =========================
    # 8. Account Age in Years
    # =========================
    # Customer stability indicator

    df['account_age_years'] = (
        df['account_age_months'] / 12
    )

    # =========================
    # 9. Credit Score Buckets
    # =========================
    # Common financial risk segmentation

    df['credit_score_band'] = pd.cut(
        df['credit_score'],
        bins=[300, 580, 670, 740, 800, 900],
        labels=['Poor', 'Fair', 'Good', 'Very_Good', 'Excellent']
    )

    # =========================
    # 10. Income-Credit Interaction
    # =========================
    # Combines earning power with creditworthiness

    df['income_credit_interaction'] = (
        df['annual_income'] * df['credit_score']
    )
    features = ['debt_to_income_ratio','loan_to_income_ratio','debt_per_loan','late_payment_ratio','spending_balance_ratio','digital_activity_score','total_transaction_activity','account_age_years','credit_score_band','income_credit_interaction']
    return df,features

