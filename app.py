"""ML model is hosted in this flask app"""

import math

from flask import Flask, request, json, render_template
import pandas as pd
import numpy as np
import pickle

# naming our app as app
app = Flask(__name__)

# loading the pickle file for creating the web app
model = pickle.load(open("./models/finalized_model.sav", "rb"))
scalerModel = pickle.load(open("./models/scaler_model.sav", "rb"))

# reading and massaging the datasets
enDf = pd.read_csv('UnbaisedTestDataset1.csv')
X = enDf.loc[:, [X for X in enDf.columns if X != 'OverallRating']]

X['ProductId'] = X['ProductId'].astype('category')
X['BuyerId'] = X['BuyerId'].astype('category')
X['SupplierId'] = X['SupplierId'].astype('category')
X['TimelyDeliveries'] = X['TimelyDeliveries'].astype('category')
X['ProductQuality'] = X['ProductQuality'].astype('category')
X['ValueForMoney'] = X['ValueForMoney'].astype('category')

X['PercReturns'] = X['NumberOfReturns'] / X['ProductQuantity']
X.drop(['NumberOfReturns', 'ProductQuantity', 'DateOfPurchase', 'PURCHASE_OPRDER'], axis=1, inplace=True)

X_categorical = X.select_dtypes(include=['category'])
X_dummies = pd.get_dummies(X_categorical, drop_first=True)
X.drop(X_categorical.columns, axis=1, inplace=True)
X = pd.concat([X, X_dummies], axis=1)
X.drop(X.index, axis=0, inplace=True)


@app.route("/predictRandomForest", methods=["GET"])
def predictRandomForest():
    global X
    pid = int(request.args.get('Pid'))
    bid = int(request.args.get('Bid'))
    quantity = int(request.args.get('Qnt'))

    agDf = pd.read_csv('UBDSaggr2.csv')
    if pid is not None:
        agDf = agDf.loc[agDf['ProductId'] == pid]
        supplier_ids = agDf['SupplierId'].values

        supp_dict = {}
        for i, supplier_id in enumerate(supplier_ids):
            X = X.append(pd.Series(0, index=X.columns), ignore_index=True)
            supp_dict[X.index[i]] = supplier_id

            # massaging for buyerId,SupplierId, ProductId
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'SupplierId' and s.split('_')[1] == str(supplier_id)]] = 1
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'BuyerId' and s.split('_')[1] == str(bid)]] = 1
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'ProductId' and s.split('_')[1] == str(pid)]] = 1

            # Massaging for avg(TimelyDeliveries), avg(ProductQuality), avg(ValueForMoney)
            avgTd = math.ceil(agDf.loc[agDf['SupplierId'] == supplier_id, ['avg(TimelyDeliveries)']].values[0][0])
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'TimelyDeliveries' and s.split('_')[1] == str(avgTd)]] = 1

            avgPdq = math.ceil(agDf.loc[agDf['SupplierId'] == supplier_id, ['avg(ProductQuality)']].values[0][0])
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'ProductQuality' and s.split('_')[1] == str(avgPdq)]] = 1

            avgVfm = math.ceil(agDf.loc[agDf['SupplierId'] == supplier_id, ['avg(ValueForMoney)']].values[0][0])
            X.loc[i, [s for s in X.columns if
                      '_' in s and s.split('_')[0] == 'ValueForMoney' and s.split('_')[1] == str(avgVfm)]] = 1

            # massaging for PricingPerUnit, PercReturns
            X.loc[i, ['PricingPerUnit']] = agDf.loc[agDf['SupplierId'] == supplier_id, ['avg(PricingPerUnit)']].values[0][0]
            X.loc[i, ['PercReturns']] = agDf.loc[agDf['SupplierId'] == supplier_id, ['avg(NumberOfReturns)']].values[0][
                                            0] / quantity

        X.loc[:, ['PricingPerUnit']] = scalerModel.transform((X.loc[:, ['PricingPerUnit']]))
        y = model.predict(X)
        pred = pd.DataFrame(np.column_stack([X.iloc[:, 0], y]), columns=['price', 'rate'])
        pred.sort_values(by=['rate'], ascending=False, inplace=True)
        top10 = pred.head(10)

        top10List = []
        for row in top10.index:
            supp_id = supp_dict[row]
            pricing = agDf.loc[agDf['SupplierId'] == supp_id, ['avg(PricingPerUnit)']].values[0][0]
            top10List.append([str(supp_id), str(pricing)])

        return json.dumps(top10List)


if __name__ == "__main__":
    app.run(debug=True)
