import numpy as np
import pandas as pd
from sklearn import linear_model
import math

if __name__ == '__main__':
    df = pd.read_csv("../data/homeprices_multiple.csv")

    # Tính giá trị trung bình của dữ liệu
    median_bedrooms = math.floor(df['bedrooms'].median())

    # Gán giá trị trung bình dữ liệu vào các chỗ missing value
    df['bedrooms'] = df['bedrooms'].fillna(median_bedrooms)

    # Sử dụng mô hình LinearRegression
    reg = linear_model.LinearRegression()

    # Fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    reg.fit(df[['area', 'bedrooms', 'age']], df['price'])

    # Các hệ số góc sau khi huấn luyện mô hình
    m = reg.coef_
    print(f"Các hệ số góc (m): {m}")
    b = reg.intercept_
    # Hệ số chặn sau khi huấn luyện mô hình
    print(f"Hệ số chặn (b): {b}")

    # Dự đoán prices nhà dựa trên [area, bedrooms, age] sau khi có mô hình dự đoán
    areas, bedrooms, age = [3000, 2500], [3, 4], [40, 5]
    # prices = reg.predict([[areas, bedrooms, age]])
    prices1 = m[0] * areas[0] + m[1] * bedrooms[0] + m[2] * age[0] + b
    prices2 = m[0] * areas[1] + m[1] * bedrooms[1] + m[2] * age[1] + b
    print(f"Với {areas[0]}, {bedrooms[0]}, {age[0]} giá là {prices1}")
    print(f"Với {areas[1]}, {bedrooms[1]}, {age[1]} giá là {prices2}")