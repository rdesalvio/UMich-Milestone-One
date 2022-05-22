import xgboost as xgb
from hyperopt import fmin, STATUS_OK, STATUS_FAIL, Trials, hp, tpe
from sklearn.metrics import auc, log_loss, roc_auc_score, brier_score_loss


def brier_skill_score(y, yhat, brier_ref):
    # calculate the brier score
    bs = brier_score_loss(y, yhat)
    res = 1.0 - (bs / brier_ref)
    return res if res > 0 else 1


probabilities = [0.0 for _ in range(len(y_train))]
avg_brier = brier_score_loss(y_train, probabilities)


def objective(space):
    clf = xgb.XGBClassifier(
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        n_estimators=space['n_estimators'],
        colsample_bytree=space['colsample_bytree'],
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=int(space['reg_lambda']),
        min_child_weight=int(space['min_child_weight']),
        max_delta_step=int(space['max_delta_step'])
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict_proba(X_test)[:, 1]
    accuracy = brier_skill_score(y_test, pred, avg_brier)
    print("SCORE:", accuracy)
    return {'loss': accuracy, 'status': STATUS_OK}


def build_xg_boost(X_train, X_test, y_train, y_test):
    """
    This section is for optimization. optimization class will user hyperoptimzation to find best parameters
    """

    """
    opimization = Optimization(X_train, y_train, X_test, y_test)
    params = opimization.run()
    print(params)
    """

    space = {
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.25, 0.01),
        'max_depth': hp.choice('max_depth', np.arange(3, 50, dtype=int)),
        'n_estimators': hp.choice("n_estimators", np.arange(100, 300, 20, dtype=int)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.uniform('gamma', 1, 9),
        'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_delta_step': hp.quniform('max_delta_step', 1, 10, 1)
    }

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)

    print(best_hyperparams)

    clf = xgb.XGBClassifier(
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        n_estimators=space['n_estimators'],
        colsample_bytree=space['colsample_bytree'],
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=int(space['reg_lambda']),
        min_child_weight=int(space['min_child_weight']),
        max_delta_step=int(space['max_delta_step'])
    )

    model.fit(X_train, y_train)
    # save model to local variable for ease of reuse
    pickle.dump(model, open(xgboost_filename, 'wb'))

    return model


if os.path.exists(xgboost_filename) != True:
    xgboost_model = build_xg_boost(X_train, X_test, y_train, y_test)
else:
    xgboost_model = pickle.load(open(xgboost_filename, 'rb'))