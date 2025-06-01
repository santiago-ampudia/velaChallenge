# Final Evaluation Report

**Test Set Size:** 1,000
**Positives (Success):** 100
**Negatives (Failure):** 900

## Overall Metrics
- True Positives (TP):  42
- False Positives (FP): 121
- False Negatives (FN): 58
- True Negatives (TN):  779
- **Precision:** 0.258
- **Recall:**    0.420
- **F1 Score:**  0.319

## Calibrated Rules Detail

| # | Feature | Type | Threshold | Rule Prec. | Rule Support | Comb. Prec. | Comb. Support | Examples (Descriptive CoT excerpts) |
|---|---|---|---|---|---|---|---|---|
| 1 | education_institution | ordinal | 4 | 0.200 | 194 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Jerry Coleman's success is largely due to his extensive experience of over 20 years coupled with high emotional intelligence, which likely enhanced his leadership effectiveness. His prior CEO exper..."

| 2 | number_of_leadership_roles | continuous | 2 | 0.109 | 498 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "James Burgess's success is likely due to his high educational qualifications, significant industry achievements, and previous startup funding experience as a CEO. His significant press coverage and..."
&nbsp;&nbsp;&nbsp;&nbsp;- "Ilya Volodarsky's success at Segment is likely influenced by his CEO experience and high educational qualifications, aligning with the company's tech focus. His experience in NASDAQ-listed companie..."

| 3 | yoe | continuous | 19.542 | 0.102 | 218 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Angela Koehler's success can be attributed to her extensive experience (over 23 years), high educational qualifications, and CEO experience. Her technical leadership roles and career growth indicat..."
&nbsp;&nbsp;&nbsp;&nbsp;- "Tjeerd Barf's success can be attributed to his high educational qualifications and 20 years of experience. His technical leadership roles and experience in multiple companies suggest a strong under..."

| 4 | emotional_intelligence | continuous | 0 | 0.090 | 700 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Craig Evans' success, with 16 years of experience and high educational qualifications, is likely due to his technical leadership roles and involvement in five companies. His high emotional intellig..."

| 5 | technical_leadership_roles | continuous | True | 0.135 | 390 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 6 | ceo_experience | binary | True | 0.089 | 294 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Chris Pihl's success can be attributed to his extensive experience and high educational qualifications. His technical leadership roles and big company experience have provided him with a strong fou..."

| 7 | perseverance | ordinal | 2 | 0.105 | 593 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 8 | nasdaq_company_experience | binary | True | 0.178 | 303 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Rick Chin's success is likely due to his extensive experience and high educational qualifications. His involvement in multiple companies and NASDAQ company experience have likely provided him with ..."
&nbsp;&nbsp;&nbsp;&nbsp;- "Tracy Sun's success can be attributed to her significant years of experience (over 16 years) and big company exposure, which likely provided her with robust industry knowledge and operational skill..."

| 9 | significant_press_media_coverage | ordinal | True | 0.179 | 173 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Chris Pihl's success can be attributed to his extensive experience and high educational qualifications. His technical leadership roles and big company experience have provided him with a strong fou..."
&nbsp;&nbsp;&nbsp;&nbsp;- "Justin Call's success, supported by his high educational qualification and experience in big companies, suggests a strong foundation in handling business operations. His technical leadership roles ..."

| 10 | board_advisor_roles | continuous | True | 0.192 | 290 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Dan Carroll's relatively short experience of about 6.3 years is offset by his high educational qualifications and significant press coverage. His technical leadership roles and experience in a NASD..."

| 11 | number_of_companies | continuous | 20.000 | 0.667 | 2 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 12 | VC_experience | binary | True | 0.260 | 124 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Arad Levertov's success is likely influenced by his high emotional intelligence, extensive NASDAQ company experience, and significant leadership roles. His medium-high educational field of study an..."

| 13 | extroversion | binary | True | 0.100 | 207 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Craig Evans' success, with 16 years of experience and high educational qualifications, is likely due to his technical leadership roles and involvement in five companies. His high emotional intellig..."

| 14 | persona | ordinal | ['L2_1', 'L2_2', 'L3_3', 'L3_6'] | 1.000 | 2 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 15 | career_growth | continuous | True | 0.102 | 650 | 0.229 | 290 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*
