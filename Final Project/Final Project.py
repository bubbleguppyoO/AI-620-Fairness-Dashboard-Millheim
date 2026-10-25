##Final project
##Created 10/13/26 by Tori Millheim

import pandas as pd
import zipfile
import joblib

#loading in required files
hd = pd.read_csv("hd2023.csv")
adm = pd.read_csv("adm2023_RV.csv")
sfa = pd.read_csv("sfa2223_RV.csv")
gr = pd.read_csv("gr2023_RV.csv")

#inspection of data found in cvs
for name, df in [("HD", hd), ("ADM", adm), ("SFA", sfa), ("GR", gr)]:
    print(name, df.shape)
    print("UNITID present:", "UNITID" in df.columns)
    print(df.head(2), "\n")

#analyzing the dataset that is common across all csv files
common = set(hd["UNITID"]) & set(adm["UNITID"]) & set(sfa["UNITID"]) & set(gr["UNITID"])
print("Schools in all four files:", len(common))

#PELL Grant
for term in ["grt", "fed", "pg"]:
    m = [c for c in sfa.columns if term in c.lower()]
    #print(term, "->", m[:12])

#print(sfa["UPGRNTP"].describe())
#print("missing:", sfa["UPGRNTP"].isna().sum())

#Dictionary check
with zipfile.ZipFile('GR2023_Dict.zip') as z:
    z.extractall('.')
    print(z.namelist())

xls = pd.ExcelFile('gr2023.xlsx')

#Finding GRTYPE and values 
varlist = pd.read_excel('gr2023.xlsx', sheet_name='Varlist')
mask = varlist.apply(lambda col: col.astype(str).str.contains('GRTYPE', na=False)).any(axis=1)
#print(varlist[mask].to_string())

desc = pd.read_excel('gr2023.xlsx', sheet_name='Description')
mask2 = desc.apply(lambda col: col.astype(str).str.contains('GRTYPE', na=False)).any(axis=1)
#print(desc[mask2].to_string())


desc = pd.read_excel('gr2023.xlsx', sheet_name='Description')
print("Columns:", desc.columns.tolist())
print("Number of GRTYPE rows:", (desc['varName'] == 'GRTYPE').sum())
print(desc[desc['varName'] == 'GRTYPE'].to_string())

freq = pd.read_excel('gr2023.xlsx', sheet_name='Frequencies')
print("Columns:", freq.columns.tolist())
mask = freq.apply(lambda col: col.astype(str).str.contains('GRTYPE', na=False)).any(axis=1)
print(freq[mask].to_string())

#gr = pd.read_excel('gr2023.xlsx')

gr_bach = gr[gr['GRTYPE'] == 6].copy()
print(gr_bach.shape)
print("Unique UNITIDs:", gr_bach['UNITID'].nunique())
print([c for c in gr.columns if 'RT' in c.upper()])

print(gr.columns.tolist())

cohort = gr[gr['GRTYPE'] == 8][['UNITID', 'GRTOTLT']].rename(columns={'GRTOTLT': 'adj_cohort'})
compl = gr[gr['GRTYPE'] == 12][['UNITID', 'GRTOTLT']].rename(columns={'GRTOTLT': 'completers'})

rates = cohort.merge(compl, on='UNITID', how='inner')
rates['grad_rate'] = rates['completers'] / rates['adj_cohort']
print(rates.shape)
print(rates['grad_rate'].describe())

merged = rates.merge(hd, on='UNITID', how='left')
print("After HD:", merged.shape)

merged = merged.merge(adm, on='UNITID', how='left')
print("After ADM:", merged.shape)

sfa_pell = sfa[['UNITID', 'UPGRNTP', 'UPGRNTN']]
merged = merged.merge(sfa_pell, on='UNITID', how='left')
print("After SFA:", merged.shape)

merged['accept_rate'] = merged['ADMSSN'] / merged['APPLCN']
print(merged[['grad_rate', 'accept_rate', 'UPGRNTP']].isna().sum())

## dropping schools with no reported Pell %
merged = merged[merged['UPGRNTP'].notna()].copy()

##filling in school admissions rates based on median
merged['accept_rate'] = merged['accept_rate'].fillna(merged['accept_rate'].median())

##high/low graduation rates
threshold = merged['grad_rate'].median()
merged['high_grad'] = (merged['grad_rate'] >= threshold).astype(int)
print("Threshold:", threshold)
print(merged['high_grad'].value_counts())

##data Analysis
##key distributions
print(merged[['grad_rate', 'accept_rate', 'UPGRNTP']].describe())

##define the Pell groups and gather graduation rates
merged['pell_group'] = pd.qcut(merged['UPGRNTP'], 3, labels=['low', 'medium', 'high'])
print(merged['pell_group'].value_counts())

print(merged.groupby('pell_group', observed=True)['grad_rate'].agg(['mean', 'median', 'count']))

##AI MODEL TRAINING
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

features = ['accept_rate', 'UPGRNTP']
X = merged[features]
y = merged['high_grad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Overall accuracy:", accuracy_score(y_test, y_pred))

#checking accuracy across all three Pell groups
from sklearn.metrics import recall_score

test_groups = merged.loc[X_test.index, 'pell_group']

for grp in ['low', 'medium', 'high']:
    mask = test_groups == grp
    acc = accuracy_score(y_test[mask], y_pred[mask])
    rec = recall_score(y_test[mask], y_pred[mask])
    print(f"{grp}: n={mask.sum()}, accuracy={acc:.3f}, recall={rec:.3f}")

merged.to_csv('merged_ipeds.csv', index=False)

joblib.dump(model, 'logreg_model.pkl')
print("AUDIT")
##Audit
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import MetricFrame

merged = pd.read_csv('merged_ipeds.csv')
model = joblib.load('logreg_model.pkl')

features = ['accept_rate', 'UPGRNTP']
X = merged[features]
y = merged['high_grad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model.predict(X_test)
sensitive = merged.loc[X_test.index, 'pell_group']

mf = MetricFrame(
    metrics={'accuracy': accuracy_score, 'recall': recall_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive
)
print(mf.by_group)
print("Recall disparity (max - min):", mf.difference()['recall'])
print("Accuracy disparity (max - min):", mf.difference()['accuracy'])
print("Recall ratio (min / max):", mf.ratio()['recall'])

mask = sensitive == 'high'
print("High-Pell test positives:", y_test[mask].sum(), "of", mask.sum())
