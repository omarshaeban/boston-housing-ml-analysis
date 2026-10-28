import numpy as np
import pandas as pd
from analysis import load_data,regress,classify


def test_all_seven_scenarios(tmp_path):
    rng=np.random.default_rng(42)
    n=110
    a=rng.normal(4,2,n)
    # Make two correlated predictors and a mixture of target values.
    medv=27+5*a+rng.normal(0,8,n)
    file=tmp_path/'BostonHousing.csv'
    pd.DataFrame({'CRIM':a,'ZN':a*1.7+rng.normal(0,.02,n),'TAX':rng.integers(200,400,n),
                  'NOX':rng.normal(.5,.1,n),'RM':rng.normal(6,1,n),'MEDV':medv}).to_csv(file,index=False)
    f=load_data(file)
    reg=regress(f)
    cl=classify(f)
    assert list(reg.task)==['A_drop_TAX','B_drop_TAX_and_highly_correlated','C_PCA_unscaled_95pct','D_PCA_scaled_95pct']
    assert len(cl)==3 and cl.accuracy.between(0,1).all()
    assert reg.R2.notna().all()
