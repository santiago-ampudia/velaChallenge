# Final Evaluation Report

**Test Set Size:** 1,000
**Positives (Success):** 100
**Negatives (Failure):** 900

## Overall Metrics
- True Positives (TP):  38
- False Positives (FP): 103
- False Negatives (FN): 62
- True Negatives (TN):  797
- **Precision:** 0.270
- **Recall:**    0.380
- **F1 Score:**  0.315

## Calibrated Rules Detail

| # | Feature | Type | Threshold | Rule Prec. | Rule Support | Comb. Prec. | Comb. Support | Examples (Descriptive CoT excerpts) |
|---|---|---|---|---|---|---|---|---|
| 1 | education_institution | ordinal | 4 | 0.200 | 194 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Jerry Coleman's success is largely due to his extensive experience of over 20 years coupled with high emotional intelligence, which likely enhanced his leadership effectiveness. His prior CEO exper..."

| 2 | number_of_leadership_roles | continuous | 2 | 0.109 | 498 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "James Burgess's success is likely due to his high educational qualifications, significant industry achievements, and previous startup funding experience as a CEO. His significant press coverage and..."

| 3 | yoe | continuous | 19.542 | 0.102 | 218 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 4 | emotional_intelligence | continuous | 0 | 0.090 | 700 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Craig Evans' success, with 16 years of experience and high educational qualifications, is likely due to his technical leadership roles and involvement in five companies. His high emotional intellig..."

| 5 | ceo_experience | binary | True | 0.089 | 294 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Chris Pihl's success can be attributed to his extensive experience and high educational qualifications. His technical leadership roles and big company experience have provided him with a strong fou..."

| 6 | technical_leadership_roles | continuous | True | 0.135 | 390 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 7 | perseverance | ordinal | 2 | 0.105 | 593 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Arad Levertov's success is likely influenced by his high emotional intelligence, extensive NASDAQ company experience, and significant leadership roles. His medium-high educational field of study an..."
&nbsp;&nbsp;&nbsp;&nbsp;- "Rick Chin's success is likely due to his extensive experience and high educational qualifications. His involvement in multiple companies and NASDAQ company experience have likely provided him with ..."

| 8 | nasdaq_company_experience | binary | True | 0.178 | 303 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- "Jerry Coleman's success is largely due to his extensive experience of over 20 years coupled with high emotional intelligence, which likely enhanced his leadership effectiveness. His prior CEO exper..."

| 9 | significant_press_media_coverage | ordinal | True | 0.179 | 173 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*

| 10 | number_of_companies | continuous | 20.000 | 0.667 | 2 | 0.218 | 226 |  |
&nbsp;&nbsp;&nbsp;&nbsp;- *No positive examples found*
