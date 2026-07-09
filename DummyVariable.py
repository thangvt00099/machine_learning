import pandas as pd
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    df = pd.read_csv("data/dummy_onehotencoding.csv")

    # Tạo các cột dummy variables (biến giả) cho các category
    dummies = pd.get_dummies(df['town'])

    # Nối cột dummy variables vào data frame
    merged = pd.concat([df, dummies], axis='columns')

    # Loại bỏ 1 cột dummy variables tránh dummy variables trap vì 1 cột có thể
    # được suy luận từ những cột còn lại
    final = merged.drop(['town', 'west windsor'], axis='columns')
    print(final)

    model = LinearRegression()

    # Tệp X_train - dữ liệu đầu vào
    X = final.drop(['price'], axis='columns')

    # Tệp Y_train - dữ liệu cần dự đoán để có kết quả
    Y = final['price']

    # fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    model.fit(X, Y)

    # Hệ số góc và hệ số chặn của mô hình sau huấn luyện xong
    m = model.coef_
    b = model.intercept_

    # Thử dự đoán ngôi nhà có diện tích 2800 ở robinsville
    print(model.predict([[2800, 0, 1]]))

    # Thử dự đoán ngôi nhà có diện tích 3400 ở west windsor
    print(model.predict([[3400, 0, 0]]))

    # Xem điểm đánh giá của mô hình dự đoán
    print(model.score(X, Y))