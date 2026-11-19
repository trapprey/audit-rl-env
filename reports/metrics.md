# Baseline metrics

### train

| policy | P | R | F1 | return | steps | sec/episode | invalid | truncated |
|---|---|---|---|---|---|---|---|---|
| random_explorer | 0.000 | 0.000 | 0.000 | -1.000 | 18.9 | 0.026 | 0 | 0 |
| naive_reconciler | 0.145 | 0.390 | 0.207 | -0.603 | 11.1 | 0.027 | 0 | 0 |
| flag_all | 0.000 | 0.000 | 0.000 | -1.000 | 4.0 | 0.025 | 0 | 0 |
| oracle | 0.800 | 0.800 | 1.000 | 1.000 | 1.0 | 0.022 | 0 | 0 |
| scripted_auditor | 0.800 | 0.800 | 1.000 | 0.975 | 13.7 | 0.027 | 0 | 0 |

Errors by type (tp/fp/fn):

| policy | duplicate_invoice | payment_allocation | fx_rate | period_cutoff |
|---|---|---|---|---|
| random_explorer | 0/3/10 | 0/1/12 | 0/1/16 | 0/4/7 |
| naive_reconciler | 10/54/0 | 3/33/9 | 0/0/16 | 7/24/0 |
| flag_all | 0/80/10 | 0/80/12 | 0/70/16 | 0/70/7 |
| oracle | 10/0/0 | 12/0/0 | 16/0/0 | 7/0/0 |
| scripted_auditor | 10/0/0 | 12/0/0 | 16/0/0 | 7/0/0 |

### dev

| policy | P | R | F1 | return | steps | sec/episode | invalid | truncated |
|---|---|---|---|---|---|---|---|---|
| random_explorer | 0.000 | 0.000 | 0.000 | -1.000 | 16.5 | 0.023 | 0 | 0 |
| naive_reconciler | 0.155 | 0.435 | 0.216 | -0.583 | 10.8 | 0.025 | 0 | 0 |
| flag_all | 0.000 | 0.000 | 0.000 | -1.000 | 4.0 | 0.024 | 0 | 0 |
| oracle | 0.800 | 0.800 | 1.000 | 1.000 | 1.0 | 0.022 | 0 | 0 |
| scripted_auditor | 0.800 | 0.800 | 1.000 | 0.973 | 14.6 | 0.028 | 0 | 0 |

Errors by type (tp/fp/fn):

| policy | duplicate_invoice | payment_allocation | fx_rate | period_cutoff |
|---|---|---|---|---|
| random_explorer | 0/7/10 | 0/4/11 | 0/2/14 | 0/4/8 |
| naive_reconciler | 9/58/1 | 4/29/7 | 0/0/14 | 8/25/0 |
| flag_all | 0/80/10 | 0/80/11 | 0/70/14 | 0/70/8 |
| oracle | 10/0/0 | 11/0/0 | 14/0/0 | 8/0/0 |
| scripted_auditor | 10/0/0 | 11/0/0 | 14/0/0 | 8/0/0 |
