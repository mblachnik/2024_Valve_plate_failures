import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import sklearn.ensemble as ens
import sklearn.metrics as me
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from external import RotationForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.preprocessing as preproc
import sklearn.neighbors as knn
import pickle
font = {
    #'family' : 'normal',
    #'weight' : 'bold',
    'size'   : 14}
plt.rc('font', **font)
feat_map = {
             'Pressure - leak line':"$P_{leak}$",
             'Temperature - leak line':"$T_{leak}$",
             'Pressure - output':"$P_{out}$",
             'Temperature - suction line':"$ T_{suct.}$",
             'Temperature - output':"$T_{out}$",
             'Flow - leak line':"$F_{leak}$",
             'Flow - output':"$F_{out}$",
             'Temp. diff':"$T_{diff}$",
            'T':'T'
}

def plot_attributes(df, cols, y=None, yp=None):
    for col in cols:

        if y is None:
            idx = np.ones((df.shape[0],), dtype=bool)
        else:
            idx = y!=yp
        x = np.arange(idx.shape[0])
        y = df.loc[idx,col]

        plt.figure()
        plt.plot(x[idx], y,'.')
        y = df.loc[~idx, col]
        plt.plot(x[~idx], y,'.')
        plt.title(col)
        plt.show()

def addAutoregression(X,lag=2):
    Xls = []
    new_cols = []
    idx = np.ones((X.shape[0]-lag,),dtype=bool)
    for j in range(lag + 1):
        if j == 0:
            Xl = X.iloc[lag - j:, :]
        else:
            Xl = X.iloc[lag - j:-j, :]
        Xl.columns = [col + f"-{j}" for col in Xl.columns]
        Xls.append(Xl)
        if j>0:
            idx &= (Xls[0].index - Xls[j].index)==pd.Timedelta( seconds=j)
        new_cols += list(Xl.columns)
    Xls = [X.reset_index(drop=True) for X in Xls]
    newX = pd.concat(Xls, axis=1, ignore_index=True)
    newX.columns = new_cols
    newX = newX.loc[idx,:]
    return newX, idx

def prepareData(data, cols_x, col_y, lag):
    X = data.loc[:, cols_x]
    X, idx = addAutoregression(X, lag)
    y = data.loc[:, col_y]
    y = y.iloc[lag:]
    y = y.loc[idx]
    return X,y

def evaluateModel(model,X,y, columns):
    yp = model.predict(X)
    print(me.accuracy_score(y_true=y, y_pred=yp))

    cm = confusion_matrix(y, yp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, )
    disp.plot()
    plt.show()
    idCor = y == yp

    for j,(i,col) in enumerate(columns):
        plt.figure(10+1)
        x = np.linspace(0, idCor.shape[0], idCor.shape[0])
        y = X[:, i]
        plt.plot(x[idCor], y[idCor], '.b')
        plt.plot(x[~idCor], y[~idCor], '.r')
        plt.ylabel(columns[j])
        plt.show()

    return yp

def evaluateFeatureImportances(model,X,y, columns):
    result = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=1, scoring='accuracy'
    )

    sorted_importances_idx = result.importances_mean.argsort()
    print(sorted_importances_idx)

    importances = pd.DataFrame(
        result.importances[sorted_importances_idx].T,
        columns=[feat_map[c] for c in columns[sorted_importances_idx]],
    )
    ax = importances.plot.box(vert=False, whis=10)
    #ax.set_title("Permutation Importances")
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in accuracy score"
                  #,fontsize=16
    )
    ax.figure.tight_layout()
    plt.show()

# %% adaa
fNames = ['dane_OT.csv',
          'dane_UT1.csv',
          'dane_UT2.csv',
          'dane_UT3.csv']
dfs = []
for f in fNames:
    df = pd.read_csv('data/' + f)
    df.Czas = pd.to_datetime(df.Czas)
    df.set_index(df.Czas, inplace=True)
    # idx = (0.03<df['Pressure - leak line']).values
    # idx &= (df['Pressure - leak line']< 0.3).values
    # idx &= (0.33<df['Temperature - leak line']).values
    # idx &= (df['Temperature - leak line']<47).values
    # idx &= (10<df[ 'Pressure - output']).values
    # idx &= (df['Pressure - output'] < 250).values
    # ss =df.shape[0]
    # df = df[idx]
    # print(f" {ss} => {df.shape[0]}")
    dfs.append(df)

cols =  ['Czas2', #0
         'Czas', #1
         'Pressure - leak line', #2
         'Temperature - leak line', #3
         'Pressure - output', #4
         'Temperature - suction line', #5
        'Temperature - output', #6
         'Flow - leak line', #7
         'Flow - output',#8
        'Temp. diff', #9
         'stan']

cols_x = ['Pressure - leak line', #2
         'Temperature - leak line', #3
         'Pressure - output', #4
         'Temperature - suction line', #5
        'Temperature - output', #6
         'Flow - leak line', #7
         'Flow - output',#8
        'Temp. diff', #9
         ]
col_y =  "stan"

dfs[0] = dfs[0].iloc[np.r_[0:2447,2449:dfs[0].shape[0]]] #Pozbywamy się przypadku z błędną temperaturą=0
# %%

# df = df.iloc[np.r_[0:2447,2449:df.shape[0]]]
data_UT2 = pd.concat([
            dfs[0].iloc[-15000:,:],
            dfs[2],
        ],axis=0)

data_UT3 = pd.concat([
            dfs[0].iloc[-15000:,:],
            dfs[3],
        ], axis=0)

data_UT1 = pd.concat([
            dfs[0].iloc[:-15000,:],
            dfs[1],
        ], axis=0)
data_UT1 = data_UT1.loc[~np.any(data_UT1.isna(), axis=1), :]


# %%
n_splits=10
do_cv = True
do_grid_search = False

if do_cv:
    lag = 0
    cv = ms.StratifiedKFold(n_splits=n_splits)
    #model = ens.RandomForestClassifier(n_estimators=100, max_depth=7, max_features=0.3, n_jobs=6)
    model = MLPClassifier(hidden_layer_sizes=(100,10), learning_rate_init=0.001, max_iter=500,random_state=42)
    # model=RotationForestClassifier(n_estimators=100,max_depth=7, max_features=0.3,n_jobs=6)
    pm = preproc.StandardScaler()

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.fit_transform(X_UT1)
    res = ms.cross_val_score(model, X_UT1, y_UT1, cv=cv, n_jobs=n_splits)
    print(res)
    print( f"{np.mean(res)} +- {np.std(res)}")

if do_grid_search:
    print("Start GridSearch")
    cv = ms.StratifiedKFold(n_splits=n_splits)
    models = [
        # ('RandomForest',
        #       ens.RandomForestClassifier(n_jobs=6),
        #      {"n_estimators":[50,100,200,300],
        #       "max_depth":[5,7,9,12],
        #       "max_features":[0.3,0.5,0.6]}
        #     ),
        # ('RotationForest',
        #      RotationForestClassifier(n_jobs=6),
        #      {"n_estimators": [50, 100, 200, 300],
        #       "max_depth": [5, 7, 9, 12],
        #       "n_features_per_subset": [2,3,4]}
        #  ),
        # ('GradientBoostedTrees',
        #     ens.GradientBoostingClassifier(),
        #     {"n_estimators": [50, 100, 200, 300],
        #      "max_depth": [5, 7, 9, 12],
        #      "learning_rate": [0.05, 0.1, 0.2]}
        #  ),
        # ('kNN',
        #  knn.KNeighborsClassifier(n_jobs=6,),
        #  {
        #      "n_neighbors":[1,3,5,7,9,11,15,21,29,51,71,101,201],
        #      "weights":["uniform", "distance"]
        #  }
        # ),
        ('MLP',
         MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=0.001, max_iter=200,random_state=42),
         {"hidden_layer_sizes": [(4,),(10,),(30,),(50,),(100,),(100,10)],
          #"max_iter": [100,500,1000],
          "max_iter": [500],
          "learning_rate_init": [0.001,0.0001]}
         ),
    ]
    pm = preproc.StandardScaler()

    res_bin = []
    res_dic = []

    for lag in [0]:#,1,2]:
        X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
        X_UT1 = pm.fit_transform(X_UT1)
        for name,model,param in models:
            print(f" {model} started")
            res = ms.GridSearchCV(model,param,cv=cv,n_jobs=10)
            res.fit(X_UT1,y_UT1)
            res_dic.append({
                "model":name,
                "score":res.best_score_,
                "params":str(res.best_params_),
                "lag":lag
            })
            print(f" {model} finished")
            print(f"    {res.best_score_}")
            print(f"    {res.best_params_}")
            print(f"    {lag}")
            print("-------------------------------------------")
            res_bin.append(res)
    df = pd.DataFrame(res_dic)
    df.to_csv("grid_search_results_MLP2_all.csv")
    with open("grid_search_results_MLP2_all.pickl","bw") as f:
        pickle.dump(res_bin, f)



#%%
do_UT23 = True
do_pca = False
if do_UT23:
    print("=========== Starting visualization ===========")
    from sklearn.decomposition import PCA

    lag = 0
    #model = ens.RandomForestClassifier(n_estimators=100, max_depth=7, max_features=0.3, n_jobs=6)
    model = MLPClassifier(hidden_layer_sizes=(100,10), learning_rate_init=0.001, max_iter=500, random_state=42)
    #model = knn.KNeighborsClassifier(n_neighbors=201, weights="distance", n_jobs=6)
    # model=RotationForestClassifier(n_estimators=100,max_depth=7, max_features=0.3,n_jobs=6)
    pm = preproc.StandardScaler()

    X_UT1, y_UT1 = prepareData(data_UT1, cols_x, col_y, lag)
    X_UT1 = pm.fit_transform(X_UT1)

    np.corrcoef(X_UT1.T)

    cols_x_corr = cols_x
    if do_pca: # PCA - temperatura
        print("=========== do PCA ===========")
        #attrs = np.array([False if ('Temperature - output' in c) or ('Temperature - suction line' in c)  else True for c in cols_x])
        attrs = np.array(
            [False if 'Temperature' in c else True for c in cols_x])
        cols_x_corr = [cols_x[i] for i, a in enumerate(attrs) if a] + ["T"]

        X_corr = X_UT1[:, ~attrs]
        pca = PCA(n_components=1)
        pca.fit(X_corr)
        un_corr = pca.transform(X_corr)
        X_UT1 = np.hstack((X_UT1[:,attrs],un_corr))
        #Order of attributes to display
        # cols_x_final = [col for idx,col in zip(attrs,cols_x) if idx]
        # cols_x_final.append("Temperature")
        cols_to_plot = ['Pressure - output',
                        #'T',
                        #'Flow - leak line',
                        #'Flow - output'
                        ]
    else:
        # Order of attributes to display
        #cols_x_final = cols_x
        cols_to_plot = [
                        'Pressure - output',
                        #'Temperature - output',
                         #'Flow - leak line',
                         #'Flow - output'
                        ]
    # Attribute and an id of this attribute in the dataset after feature selection
    cols_x_map = {col: i for i, col in enumerate(cols_x_corr)}
    columns = [(cols_x_map[col], col) for col in cols_to_plot]
    print("=========== Training model ===========")
    model.fit(X_UT1, y_UT1)

    X_UT2, y_UT2 = prepareData(data_UT2,cols_x,col_y,lag)
    X_UT2 = pm.transform(X_UT2)
    if do_pca:
        print("=========== Apply PCA to UT2 ===========")
        # PCA - temperatura
        X_corr = X_UT2[:, ~attrs]
        un_corr = pca.transform(X_corr)
        X_UT2 = np.hstack((X_UT2[:, attrs], un_corr))
    print("=========== Apply Model to UT2 ===========")
    yp_UT2 =evaluateModel(model,X_UT2, y_UT2, columns)
    evaluateFeatureImportances(model, X_UT2, y_UT2, np.array(cols_x_corr))

    X_UT3, y_UT3 = prepareData(data_UT3,cols_x,col_y,lag)
    X_UT3 = pm.transform(X_UT3)
    if do_pca:
        print("=========== Apply PCA to UT3 ===========")
        #PCA - temperatura
        X_corr = X_UT3[:, ~attrs]
        un_corr = pca.transform(X_corr)
        X_UT3 = np.hstack((X_UT3[:, attrs], un_corr))
    print("=========== Apply Model to UT3 ===========")
    yp_UT3 = evaluateModel(model,X_UT3, y_UT3, columns)
    evaluateFeatureImportances(model, X_UT3, y_UT3, np.array(cols_x_corr))

#%%
import math
doNoise = True
n_repeats = 20
if doNoise:
    scores_all = []
    stds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25]
    for XX,yy,lab in [(X_UT2,y_UT2, "Failure 2"),(X_UT3,y_UT3,"Failure 3")]:
        scores = []
        for std_init in stds:
            accs = []
            for r in range(n_repeats):
                dim = XX.shape
                std_vect = np.ones((1,dim[1])) * std_init
                noise = np.random.randn(dim[0],dim[1])
                noise = noise * std_vect
                XXn = XX + noise
                yyp = model.predict(XXn)
                acc = accuracy_score(y_true=yy,y_pred=yyp)
                accs.append(acc)
            accs = np.array(accs)
            acc_fin = np.mean(accs)
            std_fin = np.std(accs)

            scores.append({"acc":acc_fin,
                          "std":std_fin,
                          "noise":std_init,
                           'label':lab})
        scores_all += scores
        res = pd.DataFrame(scores)
        plt.errorbar(x=res["noise"],y=res["acc"],yerr=res["std"],label=lab)

    res_all =  pd.DataFrame(scores_all)
    plt.xlabel("Level of noise")
    plt.ylabel("Prediction accuracy")
    plt.legend()
    plt.show()




#%%
#plot_data(data_UT3.loc[:,cols_x])
#plot_data(data_UT2.loc[:,cols_x],['Temperature - output'],y_UT2,yp_UT2)



plt.scatter(data_UT2.loc[:,'Pressure - output'],data_UT2.loc[:,
		  'Flow - leak line'],c=y_UT2, alpha=0.1,)
plt.xlabel('Pressure - output')
plt.ylabel('Flow - leak line')
plt.show()

plt.scatter(data_UT3.loc[:,'Pressure - output'],data_UT3.loc[:,
		  'Flow - leak line'],c=y_UT3, alpha=0.1,)
plt.xlabel('Pressure - output')
plt.ylabel('Flow - leak line')
plt.show()