import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

def linear_regression(reg, df, area):
    # Fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    reg.fit(df[['area']], df['price'])

    # Hệ số góc | Hệ số cân bằng tính được bằng cách huấn luyện mô hình dựa
    # trên dữ liệu có sẵn
    m = reg.coef_
    b = reg.intercept_

    # Tính giá trị dự đoán được dựa trên mô hình | dùng công thức y = mx + b
    # hoặc reg.predict() -> xảy ra warning
    y = m * area + b
    return y, reg


if __name__ == '__main__':
    df = pd.read_csv("../data/homeprices.csv")
    print(df)

    # Sử dụng mô hình LinearRegression (Hồi quy tuyến tính) | y = mx + b
    reg = linear_model.LinearRegression()
    y, reg = linear_regression(reg, df, 3300)

    df_area = pd.read_csv("../data/areas.csv")
    df_area['prices'] = reg.predict(df_area[['area']])
    df_area.to_csv("../data/prediction.csv", index=False)

    # Sử dụng matplotlib vẽ biểu đồ phân tán dữ liệu - plt.scatter()
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.scatter(df['area'], df['price'], color='red', marker='o')
    plt.plot(df['area'], reg.predict(df[['area']]), color='blue')
    plt.show()

