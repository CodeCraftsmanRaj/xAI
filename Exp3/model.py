from sklearn.ensemble import RandomForestClassifier


def create_model(config):
    model_config = config["model"]

    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_depth=model_config["max_depth"],
        min_samples_split=model_config["min_samples_split"],
        min_samples_leaf=model_config["min_samples_leaf"],
        max_features=model_config["max_features"],
        n_jobs=model_config["n_jobs"],
        random_state=config["experiment"]["random_state"]
    )

    return model