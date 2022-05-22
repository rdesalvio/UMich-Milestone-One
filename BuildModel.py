import mysql.connector
import pandas as pd
import time
import os.path
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from hyperopt import fmin, STATUS_OK, STATUS_FAIL, Trials, hp, tpe
from sklearn.metrics import auc, log_loss, roc_auc_score, brier_score_loss, f1_score, average_precision_score, make_scorer
from sklearn.model_selection import KFold, cross_val_score, RepeatedKFold
import numpy as np
import MySQLdb
import MySQLdb.cursors
import wandb

wandb.init(project="my-test-project", entity="desalvio")

filename = 'finalized_model.sav'
filename_balanced = 'finalized_model_balanced.sav'
xgboost_filename = 'xg_boost_finalized.json'
calibrated_xgboost_filename = 'calibrated_xg_boost_finalized.json'

cols = ['ShotType', 'Distance', 'Angle', 'X_coordinate', 'Y_coordinate',
        'EWDiffLastEvent', 'NSDiffLastEvent', 'TimeSinceLastShiftChange', 'GameTimeSeconds',
        'PrevEventType', 'PrevEventTimeSince', 'PrevEventWasFriendly',
        'OffWing', 'Position', 'ShotsLastTen']

mydb = MySQLdb.connect(
        host="localhost", user="root", passwd="root", db="nhl",
        cursorclass=MySQLdb.cursors.SSCursor) # put the cursorclass here
mycursor = mydb.cursor()


def get_data_train():
    sql = "select * from new_shot_table nst inner join game g on g.gameid = nst.gameid where State = '5v5'"
    df = pd.read_sql(sql, mydb)
    return df


def get_data_test():
    sql = "select * from new_shot_table nst inner join game g on g.gameid = nst.gameid where State = '5v5' and g.season = '20202021'"
    df = pd.read_sql(sql, mydb)
    return df

def build_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(solver='lbfgs', max_iter=5000, class_weight='balanced').fit(X_train, y_train)


    pickle.dump(model, open(filename_balanced, 'wb'))
    return model


def clean_data(ShotData, shot_type_le=None, prev_event_type_le=None, pos_le=None, include_shotid=False):
    from sklearn import preprocessing

    # filter to just 5v5 shot attempts
    start_time = time.time()
    ShotData = ShotData[ShotData['State'] == '5v5']
    print("Compute time of filtering: {}".format(time.time() - start_time))

    start_time = time.time()
    # encode categorical value
    if shot_type_le is None:
        shot_type_le = preprocessing.LabelEncoder()
        shot_type_le.fit(ShotData['ShotType'])
    ShotData['ShotType'] = shot_type_le.transform(ShotData['ShotType'])
    print("Compute time of label encoding shot type: {}".format(time.time() - start_time))

    start_time = time.time()
    sql = "select distinct LastEventType from shottable"
    mycursor.execute(sql)
    base_types = mycursor.fetchall()
    all_base_types = [x[0] for x in base_types]
    if prev_event_type_le is None:
        prev_event_type_le = preprocessing.LabelEncoder()
        prev_event_type_le.fit(all_base_types)
    ShotData['PrevEventType'] = prev_event_type_le.transform(ShotData['PrevEventType'])
    print("Compute time of label encoding prev event type: {}".format(time.time() - start_time))

    start_time = time.time()
    if pos_le is None:
        pos_le = preprocessing.LabelEncoder()
        pos_le.fit(ShotData['Position'])
    ShotData['Position'] = pos_le.transform(ShotData['Position'])
    print("Compute time of label encoding prev event type: {}".format(time.time() - start_time))

    # convert strings to floats
    start_time = time.time()
    conv_cols = ['X_coordinate', 'Y_coordinate', 'Distance', 'Angle', 'EWDiffLastEvent', 'NSDiffLastEvent', 'TimeSinceLastShiftChange',
                 'GameTimeSeconds', 'PrevEventTimeSince', 'PrevEventDistance']
    ShotData[conv_cols] = ShotData[conv_cols].apply(pd.to_numeric, downcast="float", errors='coerce')
    print("Compute time of converting strings to floats: {}".format(time.time() - start_time))

    # convert nan to averages
    start_time = time.time()
    #ShotData.fillna(ShotData.mean())
    ShotData = ShotData.fillna(0)
    print("Compute time of filling na: {}".format(time.time() - start_time))

    """
    # Remove period and previous event distance due to high correlation with over variables
    corrMatrix = ShotData[cols].corr()
    sns.heatmap(corrMatrix, annot=True)
    plt.show()
    """
    return ShotData, shot_type_le, prev_event_type_le, pos_le


def split_data(ShotData):
    from sklearn.model_selection import train_test_split

    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(ShotData[cols], ShotData['Success'], test_size=0.30, random_state=40)
    print("Compute time of splitting: {}".format(time.time() - start_time))

    return X_train, X_test, y_train, y_test


def visualize_outputs(model, X_test, y_test):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.calibration import calibration_curve
    from sklearn import metrics

    kfold = StratifiedKFold(n_splits=10)
    results = cross_val_score(model.base_estimator, X_test, y_test, cv=kfold)
    print("Model Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))



    predictions = model.predict(X_test)
    score = model.score(X_test, y_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()

    prediction = model.predict_proba(X_test)[:, 1]
    plt.figure(figsize=(15, 7))
    plt.hist(prediction[y_test == 0], bins=50, label='Negatives')
    plt.hist(prediction[y_test == 1], bins=50, label='Positives', alpha=0.7, color='r')
    plt.xlabel('Probability of being Positive Class', fontsize=25)
    plt.ylabel('Number of records in each bucket', fontsize=25)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=25, pad=5)
    plt.show()

    fop, mpv = calibration_curve(y_test, prediction, n_bins=10, normalize=True)
    # plot perfectly calibrated
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot calibrated reliability
    plt.plot(mpv, fop, marker='.')
    plt.show()





def brier_skill_score(y_test, pred):
    # estimated brier score of a naive model always guessing 0
    brier_ref = 0.132

    # calculate the brier score
    bs = brier_score_loss(y_test, pred)
    res = 1.0 - (bs / brier_ref)

    # sign flipping is done by the make scorer method
    return res




def objective(space):
    wandb.config = {
        "learning_rate": space['learning_rate'],
        "max_depth": int(space['max_depth']),
        "n_estimators": space['n_estimators'],
        "colsample_bytree": space['colsample_bytree'],
        "gamma": space['gamma'],
        "reg_alpha":int(space['reg_alpha']),
        "reg_lambda": int(space['reg_lambda']),
        "min_child_weight": int(space['min_child_weight']),
        "max_delta_step": int(space['max_delta_step']),
        "scale_pos_weight": int(space['scale_pos_weight']),
        "eval_metric": 'logloss'
    }


    clf = xgb.XGBClassifier(
        learning_rate=space['learning_rate'],
        max_depth=int(space['max_depth']),
        n_estimators=space['n_estimators'],
        colsample_bytree=space['colsample_bytree'],
        gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']),
        reg_lambda=int(space['reg_lambda']),
        min_child_weight=int(space['min_child_weight']),
        max_delta_step=int(space['max_delta_step']),
        scale_pos_weight=int(space['scale_pos_weight']),
        eval_metric='logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        use_label_encoder=False
    )

    scorer = make_scorer(brier_skill_score, greater_is_better=False, needs_proba=True)
    num_folds = 10
    cv = RepeatedKFold(n_splits=num_folds, n_repeats=3, random_state=1)
    score = cross_val_score(clf, X_train, y_train, cv=cv, scoring=scorer, verbose=True, n_jobs=5).mean()

    """
    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="logloss",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict_proba(X_test)[:, 1]
    accuracy = brier_skill_score(y_test, pred)
    #pred = clf.predict(X_test)

    #pred = clf.predict(X_test)
    #accuracy = log_loss(y_test, pred)
    #accuracy = average_precision_score(y_test, pred) * -1
    """
    print("SCORE:", score)

    wandb.log({
        "learning_rate": space['learning_rate'],
        "max_depth": int(space['max_depth']),
        "n_estimators": space['n_estimators'],
        "colsample_bytree": space['colsample_bytree'],
        "gamma": space['gamma'],
        "reg_alpha":int(space['reg_alpha']),
        "reg_lambda": int(space['reg_lambda']),
        "min_child_weight": int(space['min_child_weight']),
        "max_delta_step": int(space['max_delta_step']),
        "scale_pos_weight": int(space['scale_pos_weight']),
        "loss": score
    })

    return {'loss': score, 'status': STATUS_OK}


def build_xg_boost(X_train, X_test, y_train, y_test):
    """
    This section is for optimization. optimization class will user hyperoptimzation to find best parameters
    """
    space = {
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.25, 0.01),
        'max_depth': hp.choice('max_depth', np.arange(3, 31, dtype=int)),
        'n_estimators': hp.choice("n_estimators", np.arange(100, 300, 20, dtype=int)),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.uniform('gamma', 1, 9),
        'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0, 1),
        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
        'max_delta_step': hp.quniform('max_delta_step', 1, 10, 1),
        'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 100, 1),
        'eval_metric': 'logloss'
    }

    trials = Trials()

    best_hyperparams = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=50,
                            trials=trials)

    print(best_hyperparams)

    xgb_classifier = xgb.XGBClassifier(
        learning_rate=best_hyperparams['learning_rate'],
        max_depth=int(best_hyperparams['max_depth']),
        n_estimators=best_hyperparams['n_estimators'],
        colsample_bytree=best_hyperparams['colsample_bytree'],
        gamma=best_hyperparams['gamma'],
        reg_alpha=int(best_hyperparams['reg_alpha']),
        reg_lambda=int(best_hyperparams['reg_lambda']),
        min_child_weight=int(best_hyperparams['min_child_weight']),
        max_delta_step=int(best_hyperparams['max_delta_step']),
        scale_pos_weight=int(best_hyperparams['scale_pos_weight']),
        eval_metric= 'logloss',
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        use_label_encoder=False
    )

    xgb_classifier.fit(X_train, y_train)

    # save model to local file for ease of reuse
    xgb_classifier.save_model(xgboost_filename)

    return xgb_classifier



def populate_xg(model, season, shot_type_encoder, last_event_encoder, state_type_encoder, type="xG"):
    #doing this because randomly got weird error. might need to actually take care of this
    pd.options.mode.chained_assignment = None  # default='warn'

    sql = "select * from game where season = " + str(season[0])
    mycursor.execute(sql)
    games = mycursor.fetchall()
    for game in games:
        sql = "select * from new_shot_table where gameid = " + str(game[0])
        df = pd.read_sql(sql, mydb)

        df, _, _, _,  = clean_data(df, shot_type_encoder, last_event_encoder, state_type_encoder)

        #create shotid list after removing unusable rows
        shot_id_list = df['ShotID'].tolist()
        df = df.drop('ShotID', 1)
        df = df.drop('Success', 1)

        probs = model.predict_proba(df[cols].to_numpy())
        i = 0
        for probability in probs:
            sqlUpdate = "UPDATE new_shot_table SET " + str(type) + " = " + str(round(probability[1], 2)) + " WHERE shotid = " + str(shot_id_list[i])
            print(sqlUpdate)
            mycursor.execute(sqlUpdate)
            mydb.commit()
            i = i + 1


df = get_data_train()
#df_test = get_data_test()
df, shot_type_le, prev_event_type_le, pos_le = clean_data(df)
#df_test, shot_type_le, prev_event_type_le, pos_le = clean_data(df_test)
X_train, X_test, y_train, y_test = split_data(df)



# Xtreme Gradient Bossting probabilities
if os.path.exists(xgboost_filename) != True:
    xgboost_model = build_xg_boost(X_train, X_test, y_train, y_test)
else:
    xgboost_model = xgb.XGBClassifier()
    xgboost_model.load_model(xgboost_filename)
    #xgboost_model = pickle.load(open(xgboost_filename, 'rb'))


feat_imp = pd.Series(xgboost_model.feature_importances_, cols).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')
plt.show()

from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(base_estimator=xgboost_model, method='sigmoid', cv='prefit')
calibrated.fit(X_test, y_test)
pickle.dump(calibrated, open(calibrated_xgboost_filename, 'wb'))

visualize_outputs(calibrated, X_test, y_test)

"""
sql = "select distinct Season from game"
mycursor.execute(sql)
seasons = mycursor.fetchall()

for season in seasons:
    print("Working on season: " + str(season[0]))
    populate_xg(calibrated, season, shot_type_le, prev_event_type_le, pos_le)

#visualize_outputs(calibrated, df_test[cols], df_test['Success'])
"""

"""
# Logistic Regression xG probabilities
if os.path.exists(filename_balanced) != True:
    model = build_model(X_train, y_train)
else:
    model = pickle.load(open(filename_balanced, 'rb'))
#visualize_outputs(model, X_test, y_test)

sql = "select distinct Season from game"
mycursor.execute(sql)
seasons = mycursor.fetchall()

for season in seasons:
    print("Working on season: " + str(season[0]))
    populate_xg(model, season, shot_type_le, prev_event_type_le, pos_le, "lr_xg")
"""