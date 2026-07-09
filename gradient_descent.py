import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math

# Dự đoán sư dụng LinearRegression
def predict_using_sklearn():
    df = pd.read_csv("data/test_scores.csv")
    # fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    reg = LinearRegression()
    reg.fit(df[['math']], df['cs'])
    return reg.coef_, reg.intercept_

def gradient_descent(x, y):
    m_curr = b_curr = 0
    n = len(x)
    # Số bước thực hiện
    iterations = 1000000
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        # Tính điểm dự đoán
        y_predicted = m_curr * x + b_curr
        # Chi phí mỗi bước (cost function)
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        # Tính đạo hàm cost function theo m
        md = -(2 / n) * sum(x * (y - y_predicted))
        # Tính đạo hàm cost function theo b
        bd = -(2 / n) * sum(y - y_predicted)
        # cập nhật m, b sau môi bước
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print("m = {} | b = {} | iteration = {} | cost = {}".format(m_curr, b_curr, i, cost))
    return m_curr, b_curr
if __name__ == '__main__':
    df = pd.read_csv(r"data/test_scores.csv")
    x = np.array(df['math'])
    y = np.array(df['cs'])
    m, b = gradient_descent(x, y)
    print("Using gradient descent function: Coef = {}, Intercept = {"
          "}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef = {}, Intercept = {}".format(m_sklearn,
                                                           b_sklearn))