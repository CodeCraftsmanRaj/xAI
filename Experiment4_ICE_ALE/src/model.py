from sklearn.ensemble import RandomForestClassifier

def build_model(cfg):
    m = cfg["model"]
    return RandomForestClassifier(
        n_estimators=m["n_estimators"],
        max_depth=m["max_depth"],
        min_samples_leaf=m["min_samples_leaf"],
        max_features=m["max_features"],
        n_jobs=m["n_jobs"],
        random_state=cfg["experiment"]["random_state"],
        class_weight="balanced"
    )
