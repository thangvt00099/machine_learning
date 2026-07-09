import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import joblib

if __name__ == '__main__':
    df = pd.read_csv("data/homeprices.csv")
    model = LinearRegression()
    model.fit(df[['area']], df['price'])
    m = model.coef_
    b = model.intercept_
    print("Coef = {}, Intercept = {}".format(m, b))
    print("Predict price of area 5000 = {}".format(model.predict([[5000]])))

    # Lưu tệp model đã được huấn luyện sử dụng pickle
    with open("data/model_pickle", "wb") as f:
        pickle.dump(model, f)

    # Sử dụng tệp model đã được huấn luyện trước đó sử dụng pickle
    with open("data/model_pickle", "rb") as f:
        mp = pickle.load(f)
    print("Predict price of area 6000 = {}".format(mp.predict([[6000]])))

    # Lưu tệp model và load lại tệp model đã được huấn luyện sử dụng joblib
    joblib.dump(model, "data/model_joblib")
    model_joblib = joblib.load("data/model_joblib")
    print("Predict price of area 7000 = {}".format(model_joblib.predict([[
        7000]])))


