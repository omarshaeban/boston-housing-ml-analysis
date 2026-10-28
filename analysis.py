"""Reconstructed BostonHousing.csv regression/classification course assignment.

No original CSV is included. Pass the file supplied with the course.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class TrainOnlyCorrelationFilter(BaseEstimator,TransformerMixin):
    """Drop redundant features using training data only."""
    def __init__(self,threshold=0.85): self.threshold=threshold
    def fit(self,X,y=None):
        x=np.asarray(X,dtype=float)
        corr=np.abs(np.corrcoef(x,rowvar=False))
        corr=np.nan_to_num(corr,nan=0)
        exclude=set()
        for j in range(corr.shape[1]):
            for k in range(j+1,corr.shape[1]):
                if corr[j,k]>self.threshold and k not in exclude:exclude.add(k)
        self.kept_=[k for k in range(x.shape[1]) if k not in exclude]
        return self
    def transform(self,X):return np.asarray(X)[:,self.kept_]


def load_data(path:str|Path)->pd.DataFrame:
    frame=pd.read_csv(path)
    frame.columns=[c.strip().upper() for c in frame.columns]
    if not {'TAX','MEDV'}.issubset(frame.columns):
        raise ValueError('BostonHousing.csv must include TAX and MEDV; received: '+', '.join(frame.columns))
    # Classification derived from MEDV, not reused as a predictor.
    features=frame.drop(columns=['MEDV','EXPENSIVE'],errors='ignore')
    for col in features.columns:
        features[col]=pd.to_numeric(features[col],errors='coerce')
    frame[features.columns]=features
    frame['MEDV']=pd.to_numeric(frame.MEDV,errors='coerce')
    frame=frame.dropna(subset=['MEDV'])
    if len(frame)<15:raise ValueError('Need at least 15 rows with valid MEDV values')
    return frame


def regress(frame:pd.DataFrame,seed=42):
    X=frame.drop(columns=['MEDV','EXPENSIVE','TAX'],errors='ignore')
    y=frame.MEDV
    train,test=train_test_split(np.arange(len(y)),test_size=.25,random_state=seed)
    methods={
        'A_drop_TAX':[],
        'B_drop_TAX_and_highly_correlated':[('corr',TrainOnlyCorrelationFilter(.85))],
        'C_PCA_unscaled_95pct':[('pca',PCA(n_components=.95,svd_solver='full'))],
        'D_PCA_scaled_95pct':[('scale',StandardScaler()),('pca',PCA(n_components=.95,svd_solver='full'))],
    }
    rows=[]
    for name,extra in methods.items():
        steps=[('impute',SimpleImputer(strategy='median'))]+extra+[('regressor',LinearRegression())]
        pipe=Pipeline(steps)
        pipe.fit(X.iloc[train],y.iloc[train])
        predicted=pipe.predict(X.iloc[test])
        pc=pipe.named_steps.get('pca')
        rows.append(dict(task=name,MAE=round(float(np.mean(np.abs(y.iloc[test]-predicted))),3),
                         MSE=round(float(mean_squared_error(y.iloc[test],predicted)),3),
                         R2=round(float(r2_score(y.iloc[test],predicted)),4),
                         retained_PCs=(int(pc.n_components_) if pc else None),
                         original_features=int(X.shape[1]),test_rows=len(test)))
    return pd.DataFrame(rows)


def classify(frame:pd.DataFrame,seed=42):
    X=frame.drop(columns=['MEDV','TAX','EXPENSIVE'],errors='ignore')
    y=(frame.MEDV>=30).astype(int)
    if y.nunique()<2 or y.value_counts().min()<3:raise ValueError('Classification needs at least three samples in each class')
    train,test=train_test_split(np.arange(len(y)),test_size=.25,random_state=seed,stratify=y)
    variants={'i_raw':[], 'ii_minmax':[('scale',MinMaxScaler())], 'iii_zscore':[('scale',StandardScaler())]}
    rows=[]
    for name,extra in variants.items():
        pipe=Pipeline([('impute',SimpleImputer(strategy='median'))]+extra+[('model',LogisticRegression(max_iter=3000,random_state=seed))])
        pipe.fit(X.iloc[train],y.iloc[train]);pred=pipe.predict(X.iloc[test])
        rows.append(dict(task=name,accuracy=round(float(accuracy_score(y.iloc[test],pred)),4),
                         precision=round(float(precision_score(y.iloc[test],pred,zero_division=0)),4),
                         recall=round(float(recall_score(y.iloc[test],pred,zero_division=0)),4),
                         f1=round(float(f1_score(y.iloc[test],pred,zero_division=0)),4),
                         test_rows=len(test)))
    return pd.DataFrame(rows)


def main():
    a=argparse.ArgumentParser(description='Boston Housing coursework reconstruction');a.add_argument('csv');a.add_argument('--output-dir',default='results');args=a.parse_args()
    data=load_data(args.csv)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    reg=regress(data);cl=classify(data)
    reg.to_csv(out/'regression_comparison.csv',index=False)
    cl.to_csv(out/'classification_comparison.csv',index=False)
    print('Regression:\n',reg.to_string(index=False),'\nClassification:\n',cl.to_string(index=False))
    print('Interpret only if input is the genuine coursework dataset.')


if __name__=='__main__':main()
