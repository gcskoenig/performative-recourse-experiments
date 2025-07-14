from recourse.scm import LinGauSeq, SCM, RFGauSeq
import pandas as pd
import kagglehub


def get_satgpa_scm():
    data = pd.read_csv('application/satgpa/satgpa.csv')
    data['y'] = data['fy_gpa']
    data = data[['sat_sum', 'hs_gpa', 'y']]
    
    seq_hs_gpa = LinGauSeq.fit(data, 'hs_gpa', [])
    seq_fy_gpa = LinGauSeq.fit(data, 'y', ['hs_gpa'])
    seq_sat_sum = LinGauSeq.fit(data, 'sat_sum', ['y'])

    seqs = [seq_hs_gpa, seq_fy_gpa, seq_sat_sum]
    satgpa_scm = SCM(seqs)
    return satgpa_scm


def get_credit_scm():
    # Download latest version
    path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")
    print("Path to dataset files:", path)
    df = pd.read_csv(path + '/UCI_Credit_Card.csv')
    df.columns
    
    df["MaxBillAmountOverLast6Months"] = df[[
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]].max(axis=1)
    df["MaxPaymentAmountOverLast6Months"] = df[[
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]].max(axis=1)
    df["MonthsWithZeroBalanceOverLast6Months"] = df[[
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]].apply(lambda row: (row == 0).sum(), axis=1)
    low_spending_threshold = 5000
    df["MonthsWithLowSpendingOverLast6Months"] = df[[
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]].apply(lambda row: (row < low_spending_threshold).sum(), axis=1)
    high_spending_threshold = 20000
    df["MonthsWithHighSpendingOverLast6Months"] = df[[
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]].apply(lambda row: (row > high_spending_threshold).sum(), axis=1)
    df["MostRecentBillAmount"] = df["BILL_AMT1"]
    df["MostRecentPaymentAmount"] = df["PAY_AMT1"]

    pay_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    df["TotalOverdueCounts"] = df[pay_columns].apply(lambda row: (row >= 1).sum(), axis=1)
    df["TotalMonthsOverdue"] = df[pay_columns].apply(lambda row: row[row >= 1].sum(), axis=1)
    df['EducationLevel'] = df['EDUCATION']
    df['y'] = df['default.payment.next.month']

    relevant_cols = ['MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months',
                     'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months',
                     'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount', 'MostRecentPaymentAmount',
                     'TotalOverdueCounts', 'TotalMonthsOverdue', 'EducationLevel', 'y']

    df_rel = df[relevant_cols].copy()
    
    # causes
    causes = ['EducationLevel', 'TotalOverdueCounts', 'TotalMonthsOverdue']
    effects = ['MaxBillAmountOverLast6Months', 'MaxPaymentAmountOverLast6Months',
               'MonthsWithZeroBalanceOverLast6Months', 'MonthsWithLowSpendingOverLast6Months',
               'MonthsWithHighSpendingOverLast6Months', 'MostRecentBillAmount', 'MostRecentPaymentAmount']
    seqs = []
    for cause in causes:
        seq = RFGauSeq.fit(df_rel, cause, [])
        seqs.append(seq)
    seqs.append(RFGauSeq.fit(df_rel, 'y', causes))
    for effect in effects:
        seq = RFGauSeq.fit(df_rel, effect, ['y'])
        seqs.append(seq)
        
    scm = SCM(seqs)
    return scm


application_scms = {
    'satgpa': get_satgpa_scm(),
    'credit': get_credit_scm()
}

shortnames_applications = {
    'satgpa': 'GPA',
    'credit': 'Credit'
}

