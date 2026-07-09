import pandas as pd
import math
from sklearn import linear_model
from word2number import w2n

def calculate_predict(m, b, experience, test_score, interview_score):
    predict = m[0] * experience + m[1] * test_score + m[2] * interview_score + b
    return predict

if __name__ == '__main__':
    df = pd.read_csv("../data/hiring.csv")

    # Xử lí missing value cột ['experience'] và convert thành number
    df['experience'] = df['experience'].fillna("zero")
    df['experience'] = df['experience'].apply(w2n.word_to_num)

    # Xử lí missing value cột test_score
    median_score = math.floor(df['test_score(out of 10)'].median())
    df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(median_score)
    print(df)

    # Sử dụng mô hình LinearRegression
    reg = linear_model.LinearRegression()

    # Fit dữ liệu (huấn luyện mô hình dựa trên dữ liệu có sẵn)
    reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out '
                                                       'of 10)']],
            df['salary($)'])

    # Hệ số góc sau khi huấn luyện mô hình
    m = reg.coef_
    # Hệ số chặn sau khi huấn luyện mô hình
    b = reg.intercept_
    print(f"Các hệ số góc (m): {m}")
    print(f"Hệ số chặn (b): {b}")

    # Dự đoán 1 vài kết quả
    experiences = [2, 1, 2, 12]
    test_score = [8, 5, 9, 10]
    interview_score =[8, 6, 6, 10]
    # predict1 = reg.predict([[experiences[0], test_score[0], interview_score[
    #     0]]])
    # predict2 = reg.predict([[experiences[1], test_score[1], interview_score[
    #     1]]])
    predict1 = calculate_predict(m, b, experiences[0], test_score[0],
                                 interview_score[0])
    predict2 = calculate_predict(m, b, experiences[1], test_score[1],
                                 interview_score[1])
    predict3 = calculate_predict(m, b, experiences[2], test_score[2],
                                 interview_score[2])
    predict4 = calculate_predict(m, b, experiences[3], test_score[3],
                                 interview_score[3])
    print(f"Với {experiences[0]} kinh nghiệm, {test_score[0]} điểm test, "
          f"{interview_score[0]} điểm phỏng vấn thì mức lương dự đoán là"
          f" {predict1}")
    print(f"Với {experiences[1]} kinh nghiệm, {test_score[1]} điểm test, "
          f"{interview_score[1]} điểm phỏng vấn thì mức lương dự đoán là"
          f" {predict2}")
    print(f"Với {experiences[2]} kinh nghiệm, {test_score[2]} điểm test, "
          f"{interview_score[2]} điểm phỏng vấn thì mức lương dự đoán là"
          f" {predict3}")
    print(f"Với {experiences[3]} kinh nghiệm, {test_score[3]} điểm test, "
          f"{interview_score[3]} điểm phỏng vấn thì mức lương dự đoán là"
          f" {predict4}")
