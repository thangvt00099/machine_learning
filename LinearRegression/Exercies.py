import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

if __name__ == '__main__':
    df = pd.read_csv("../data/canada_per_capita_income.csv")
    print(df)

    # Mô hình sử dụng (Linear Regression)
    reg = linear_model.LinearRegression()

    # Fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    reg.fit(df[['year']], df['per capita income (US$)'])

    # Hệ số góc và hệ số chặn tính được sau khi huấn luyện mô hình
    m = reg.coef_
    b = reg.intercept_
    print(f"Hệ số góc: {m}")
    print(f"Hệ số chặn: {b}")

    # Predict thu nhập bình quân ở Canada năm 2020
    # predict = reg.predict([[2020]]) # C1 dùng predict() method của mô hình
    predict = m * 2020 + b
    print(f"Thu nhập bình quân ở Canada năm 2020: {predict[0]:.2f}")

    # Biểu đồ thể hiện sự phân rã dữ liệu
    plt.title('Canada')
    plt.xlabel('Year')
    plt.ylabel('Per Capita Income (US$)')
    plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='o')
    plt.plot(df['year'], reg.predict(df[['year']]), color='blue')
    plt.show()