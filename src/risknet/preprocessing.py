from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class FoldWeightedXGBClassifier(XGBClassifier):
    """Compute imbalance weight on every fit, including inner CV folds."""
    def fit(self, X, y, **kwargs):
        self.set_params(scale_pos_weight=float((y == 0).sum() / (y == 1).sum()))
        return super().fit(X, y, **kwargs)

def make_pipeline(name, y):
    models = {
        'logistic': LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=120, class_weight='balanced', min_samples_leaf=3, n_jobs=2, random_state=42),
        'xgboost': FoldWeightedXGBClassifier(n_estimators=120, max_depth=3, learning_rate=.07, n_jobs=2, random_state=42, eval_metric='logloss')}
    return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()), ('model', models[name])])
