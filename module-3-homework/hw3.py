from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
BASE = Path(__file__).resolve().parent
df_full = pd.read_parquet(BASE / 'stocks_df_combined_2026_09_18.parquet.brotli')
# growth indicators (but not future growth)
GROWTH = [g for g in df_full.keys() if (g.find('growth_')==0)&(g.find('future')<0)]
GROWTH

# leaving only Volume ==> generate ln(Volume)
OHLCV = ['Open','High','Low','Close','Adj Close_x','Volume']

CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type']

TO_PREDICT = [g for g in df_full.keys() if (g.find('future')>=0)]
TO_PREDICT

TO_DROP = ['Year','Date','index_x', 'index_y', 'index', 'Quarter','Adj Close_y'] + CATEGORICAL + OHLCV
TO_DROP

# a small hack to avoid "divide by zero encountered in log"
df_full['ln_volume'] = df_full.Volume.replace(0, np.nan).fillna(1e-9).apply(lambda x: np.log(x))

# manually defined features
CUSTOM_NUMERICAL = ['SMA10', 'SMA20', 'growing_moving_average', 'high_minus_low_relative','volatility', 'ln_volume']

# All Supported Ta-lib indicators: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md

TECHNICAL_INDICATORS = ['adx', 'adxr', 'apo', 'aroon_1','aroon_2', 'aroonosc',
 'bop', 'cci', 'cmo','dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext',
 'macdsignal_ext', 'macdhist_ext', 'macd_fix', 'macdsignal_fix',
 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm', 'ppo',
 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk',
 'fastd', 'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr',
 'ad', 'adosc', 'obv', 'atr', 'natr', 'ht_dcperiod', 'ht_dcphase',
 'ht_phasor_inphase', 'ht_phasor_quadrature', 'ht_sine_sine', 'ht_sine_leadsine',
 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice']

TECHNICAL_PATTERNS = [g for g in df_full.keys() if g.find('cdl')>=0]
print(f'Technical patterns count = {len(TECHNICAL_PATTERNS)}, examples = {TECHNICAL_PATTERNS[0:5]}')


MACRO = ['gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
 'DGS1', 'DGS5', 'DGS10']

NUMERICAL = GROWTH + TECHNICAL_INDICATORS + TECHNICAL_PATTERNS + CUSTOM_NUMERICAL + MACRO

df = df_full.loc[df_full.Date >= '2000-01-01'].copy()
df['month_wom'] = df.Date.dt.month_name() + '_w' + ((df.Date.dt.day - 1) // 7 + 1).astype(str)
CATEGORICAL = ['Month', 'Weekday', 'Ticker', 'ticker_type', 'month_wom']
df['Weekday'] = df['Weekday'].astype(str)
df['Month'] = df['Month'].dt.month.astype(str) # Convert to string so get_dummies creates dummy columns properly
# Generate dummy variables (no need for bool, let's have int32 instead)
dummy_variables = pd.get_dummies(df[CATEGORICAL], dtype='int32')
# get dummies names in a list
DUMMIES = dummy_variables.keys().to_list()
# Concatenate the dummy variables with the original DataFrame
df_with_dummies = pd.concat([df, dummy_variables], axis=1)
def temporal_split(df, min_date, max_date, train_prop=0.7, val_prop=0.15, test_prop=0.15):
    """
    Splits a DataFrame into three buckets based on the temporal order of the 'Date' column.

    Args:
        df (DataFrame): The DataFrame to split.
        min_date (str or Timestamp): Minimum date in the DataFrame.
        max_date (str or Timestamp): Maximum date in the DataFrame.
        train_prop (float): Proportion of data for training set (default: 0.6).
        val_prop (float): Proportion of data for validation set (default: 0.2).
        test_prop (float): Proportion of data for test set (default: 0.2).

    Returns:
        DataFrame: The input DataFrame with a new column 'split' indicating the split for each row.
    """
    # Define the date intervals
    train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
    val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

    # Assign split labels based on date ranges
    split_labels = []
    for date in df['Date']:
        if date <= train_end:
            split_labels.append('train')
        elif date <= val_end:
            split_labels.append('validation')
        else:
            split_labels.append('test')

    # Add 'split' column to the DataFrame
    df['split'] = split_labels

    return df
min_date_df = df_with_dummies.Date.min()
max_date_df = df_with_dummies.Date.max()

df_with_dummies = temporal_split(df_with_dummies,
                                 min_date = min_date_df,
                                 max_date = max_date_df)
new_df = df_with_dummies.copy()
target = 'is_positive_growth_30d_future'
month_corr = new_df[[c for c in DUMMIES if c.startswith('month_wom_')]].corrwith(new_df[target]).abs().sort_values(ascending=False)
print('Q1', month_corr.head().to_string(), flush=True)
# generate manual predictions
# Let's label all prediction features with prefix "pred"
new_df['pred0_manual_cci'] = (new_df.cci>200).astype(int)
new_df['pred1_manual_prev_g1'] = (new_df.growth_30d>1).astype(int)
new_df['pred2_manual_prev_g1_and_snp'] = ((new_df['growth_30d'] > 1) & (new_df['growth_snp500_30d'] > 1)).astype(int)
new_df['pred3_manual_dgs10_5'] = ((new_df.DGS10 <= 4.5) & (new_df.DGS5 <= 4)).astype(int)
new_df['pred4_manual_dgs10_fedfunds'] = ((new_df.DGS10 > 4) & (new_df.FEDFUNDS <= 4.795)).astype(int)
hand_columns = [c for c in new_df if c.startswith('pred')]
test_mask = new_df.split.eq('test')
train_mask = ~test_mask
y_all = new_df[target].fillna(0)
X_all = new_df[NUMERICAL + DUMMIES].replace([np.inf, -np.inf], np.nan).fillna(0)
X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]
def evaluate(column):
    pred = new_df.loc[test_mask, column]
    tp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    return {'tp': tp, 'fp': fp, 'positive_predictions': tp + fp, 'precision': tp / (tp + fp) if tp + fp else None}
manual_scores = {c: evaluate(c) for c in hand_columns}
print('Manual rules', json.dumps(manual_scores), flush=True)
clf10 = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X_train, y_train)
new_df['pred5_clf_10'] = clf10.predict(X_all)
all_hand_wrong = new_df[hand_columns].ne(y_all, axis=0).all(axis=1)
new_df['only_pred5_is_correct'] = (new_df.pred5_clf_10.eq(y_all) & all_hand_wrong).astype(int)
q3 = int(new_df.loc[test_mask, 'only_pred5_is_correct'].sum())
print('Q3', q3, flush=True)
depth_scores = []
for depth in range(1, 13):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    score = float(precision_score(y_test, model.predict(X_test), zero_division=0))
    depth_scores.append({'depth': depth, 'precision': score})
    print('Depth', depth, 'precision', score, flush=True)
best = max(depth_scores, key=lambda row: row['precision'])
best_model = DecisionTreeClassifier(max_depth=best['depth'], random_state=42).fit(X_train, y_train)
new_df['pred6_clf_best'] = best_model.predict(X_all)
results = {'q1': {'feature': month_corr.index[0], 'absolute_correlation': float(month_corr.iloc[0]), 'rounded': round(float(month_corr.iloc[0]), 3)}, 'manual_scores': manual_scores, 'q2': round(max(v['precision'] for c, v in manual_scores.items() if c.startswith(('pred3', 'pred4')) and v['precision'] is not None), 3), 'q3': q3, 'q4': best['depth'], 'depth_scores': depth_scores, 'model_scores': {c: evaluate(c) for c in ['pred5_clf_10', 'pred6_clf_best']}, 'rows': len(new_df), 'dummy_count': len(DUMMIES), 'numerical_count': len(NUMERICAL), 'split_counts': new_df.split.value_counts().to_dict(), 'split_dates': {k: {'min': str(v.Date.min()), 'max': str(v.Date.max())} for k, v in new_df.groupby('split')}}
(BASE / 'homework3_results.json').write_text(json.dumps(results, indent=2))
pd.DataFrame(depth_scores).to_csv(BASE / 'homework3_depth_scores.csv', index=False)
new_df[['Date','Ticker','split',target] + hand_columns + ['pred5_clf_10','pred6_clf_best','only_pred5_is_correct']].to_parquet(BASE / 'homework3_predictions.parquet', index=False)
print(json.dumps(results, indent=2), flush=True)
