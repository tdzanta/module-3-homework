Module 3 homework (2026)
========================

Q1: 0.025.
The largest absolute correlation is for October_w4: 0.0245841676. I used (day - 1) // 7 + 1 for the week of the month, then added the month/week dummies to the model features.

Q2: 0.588.
The better new rule is pred3: DGS10 <= 4.5 and DGS5 <= 4. On the test set it has 9,351 true positives and 6,559 false positives, so precision is 9351 / 15910 = 0.5877435575. The other new rule has precision 0.5025438101.

Q3: 1,243 records.
These are test rows where the depth-10 tree predicts the true label and every hand rule, pred0 through pred4, predicts the wrong label. The tree was trained on TRAIN + VALIDATION with random_state=42.

Q4: 4.
Depth 4 has the highest test precision: 0.6291747791. It makes 6,677 positive predictions, of which 4,201 are correct. The depth-10 tree has precision 0.5912146500. I checked all depths from 1 to 12 and saved the full table in homework3_depth_scores.csv. I followed the homework's instruction to select on TEST; for a new modeling project I would choose depth on validation and reserve a separate test period for the final evaluation.

Q5:
I would add company earnings information and measures of market stress.

The first features I would try are the number of days until the next announced earnings release, the surprise from the most recently published earnings report, and recent earnings growth. Prices and technical indicators show what a stock has been doing, but they do not directly tell us whether the company's results were better or worse than expected. These features might help explain some of the cases where the hand rules and the decision tree disagree. Alpha Vantage's EARNINGS endpoint provides reported EPS, estimated EPS and earnings surprises; its EARNINGS_CALENDAR endpoint provides upcoming reports. I would check historical coverage for each ticker across India, the EU and the US rather than assume every company has the same coverage.

I would also add VIX and its recent change as measures of market stress. The same price momentum could behave differently in a calm market and during a period of uncertainty. Daily VIX data is available from FRED under VIXCLS. Since VIX reflects the US market, I would treat it as a shared global context feature and check its usefulness separately for each region.

The main thing I would be careful about is timing. An earnings surprise should only become available after the report was released, and a future earnings date should only be used if it had already been announced. For VIX, I would use the last value available before the local trading session, which matters especially for India and Europe. I would compare the expanded model with the original model using time-based validation, looking at both precision and the number of positive predictions. I would keep a separate final test period untouched rather than keep choosing features based on the same test results.

Sources: https://www.alphavantage.co/documentation/#earnings ; https://www.alphavantage.co/documentation/#earnings-calendar ; https://fred.stlouisfed.org/series/VIXCLS


Run details
-----------
Dataset: stocks_df_combined_2026_09_18.parquet.brotli
Filtered dataset: 202,292 rows from 2000-01-03 onward
Features: 184 numerical features and 115 dummy variables
Train: 136,842 rows; validation: 32,287 rows; test: 33,163 rows
Test dates: 2022-09-16 to 2026-09-18
Cleaning: replace infinity with NaN, then fill missing model inputs with 0, following the course notebook
Verification: independently recomputed precision, unique-correct cases and best depth from the saved prediction/table files
