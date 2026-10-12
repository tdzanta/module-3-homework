from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import precision_score
p=Path(__file__).resolve().parent
r=json.loads((p/'homework3_results.json').read_text())
df=pd.read_parquet(p/'homework3_predictions.parquet')
t=df.loc[df.split.eq('test')]
y=t.is_positive_growth_30d_future
hands=[c for c in t if c.startswith(('pred0','pred1','pred2','pred3','pred4'))]
unique=t.pred5_clf_10.eq(y)
for c in hands:
    unique &= t[c].ne(y)
assert int(unique.sum()) == r['q3']
assert round(precision_score(y,t.pred3_manual_dgs10_5),3)==r['q2']
scores=pd.read_csv(p/'homework3_depth_scores.csv')
assert int(scores.loc[scores.precision.idxmax(),'depth'])==r['q4']
assert abs(precision_score(y,t.pred6_clf_best)-scores.precision.max())<1e-12
assert df[['Date','Ticker']].duplicated().sum()==0
print('Verified Q2, Q3, best depth, best-model precision, and unique Date/Ticker rows.')
