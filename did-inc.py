import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf

kielmc = woo.dataWoo('kielmc')

y78 = kielmc['year'] == 1978
reg78 = smf.ols('rprice ~ nearinc', data=kielmc, subset=y78)
results78 = reg78.fit()
results78.summary()

kielmc


