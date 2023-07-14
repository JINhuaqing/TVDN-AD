from sklearn.linear_model import LogisticRegression
import numpy as np

def get_ABIC(Xs, Ys, C):
    lam = 2/C
    clf_info = LogisticRegression(penalty="l2", random_state=0, C=C);
    clf_info.fit(Xs, Ys)
    eprobs1 = clf_info.predict_proba(Xs)[:, 1];
    logL = np.sum(Ys*np.log(eprobs1) + (1-Ys)*np.log(1-eprobs1));
    
    # effective df
    fXs = np.concatenate([np.ones((Xs.shape[0], 1)), Xs], axis=-1)
    logL_der2 = -fXs.T @ np.diag(eprobs1*(1-eprobs1)) @ fXs;
    logL_der2_ridge = logL_der2 - lam * np.eye(logL_der2.shape[0]);
    dfe = np.diag(logL_der2@np.linalg.inv(logL_der2_ridge)).sum();
    aic = -2*logL + 2*dfe
    bic = -2*logL + np.log(Ys.shape[0])*dfe
    return aic, bic