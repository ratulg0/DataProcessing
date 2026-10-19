import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def quality_report(frame):
    numeric = frame.select_dtypes(include=np.number)
    return pd.DataFrame({'feature': frame.columns, 'dtype': [str(frame[c].dtype) for c in frame.columns], 'missing_count': [int(frame[c].isna().sum()) for c in frame.columns], 'missing_percent': [round(float(frame[c].isna().mean()*100),3) for c in frame.columns], 'unique_values': [int(frame[c].nunique(dropna=True)) for c in frame.columns], 'minimum': [numeric[c].min() if c in numeric else np.nan for c in frame.columns], 'maximum': [numeric[c].max() if c in numeric else np.nan for c in frame.columns]})

def descriptive_statistics(frame):
    numeric = frame.select_dtypes(include=np.number)
    out = numeric.describe(percentiles=[.05,.25,.5,.75,.95]).T
    out['missing'] = numeric.isna().sum(); out['variance'] = numeric.var(); out['skewness'] = numeric.skew(); out['iqr'] = numeric.quantile(.75)-numeric.quantile(.25)
    return out.reset_index(names='feature')

def candidate_clean(frame):
    """Impute actual NaNs for exploration; don't assume zeros mean missing."""
    out = frame.copy()
    for col in out.select_dtypes(include=np.number):
        if out[col].isna().any(): out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(exclude=np.number):
        if out[col].isna().any(): out[col] = out[col].fillna('Unknown')
    return out

def split_aware_standardise(X_train, X_test):
    """Fit StandardScaler on training rows only, then transform test rows."""
    scaler = StandardScaler()
    train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return train, test, scaler
