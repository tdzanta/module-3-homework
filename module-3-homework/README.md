Module 3 homework (2026)
========================

The solutions are in hw3.py. The answers and supporting counts are in answers_all.md.
There is also a Jupyter version in homework3_solution.ipynb.

Answers: Q1 = 0.025; Q2 = 0.588; Q3 = 1243; Q4 = 4.
The complete written answer to Q5 is in answers_all.md and the notebook.

Run
---

Use Python 3.11. Install dependencies:

    python -m pip install -r requirements.txt

Download the course dataset and put it alongside hw3.py:
https://drive.google.com/file/d/1oQSUMCs2DyQQIh8Y62UhrT00cIsE9Sr5/view
Filename: stocks_df_combined_2026_09_18.parquet.brotli

Then run:

    python hw3.py

In the original homework folder, run:

    .\.venv\Scripts\python.exe hw3.py

The script recreates homework3_results.json, homework3_depth_scores.csv and homework3_predictions.parquet.
The supplied results were calculated on the original course dataset with random_state=42.
Missing/infinite model inputs are filled with zero, matching the lecture.
The training data combines TRAIN and VALIDATION. Month/week dummies remain in all tree models.

Submission
----------

Create a public GitHub repository named module-3-homework and upload the files from this folder.
The large dataset and prediction parquet are not needed on GitHub.

Open https://courses.datatalks.club/sma-zoomcamp-2026/homework/hw03
Select the choices matching 0.025, 0.588, 1243 and depth 4.
Copy Q5 from answers_all.md. Put the actual GitHub repository URL in Homework URL.
Do not enter time spent or learning-in-public links unless they reflect your own activity.

Course assignment:
https://github.com/DataTalksClub/stock-markets-analytics-zoomcamp/blob/main/cohorts/2026/homework3.md

Status: calculations complete and verified; GitHub publication and course submission still pending.
