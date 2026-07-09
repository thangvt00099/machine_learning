import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv("data/carprices.csv")
    X = df[['Mileage', 'Age(yrs)']]
    Y = df['Sell Price($)']

    # Chia dữ liệu thành 2 tập: huấn luyện (80%) và kiểm thử (20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print("Kích thước tệp X_train sau khi chia dữ liệu: ", len(X_train))
    print("Kích thước tệp X_test sau khi chia dữ liệu: ", len(X_test))
    print("Kích thước tệp Y_train sau khi chia dữ liệu: ", len(Y_train))
    print("Kích thước tệp Y_test sau khi chia dữ liệu: ", len(Y_test))

    # Tạo model
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Hệ số góc và hệ số chặn của model
    m = model.coef_
    b = model.intercept_
    print("Coef: {} | Intercept: {}".format(m, b))

    # Kiểm tra độ chính xác của model
    print("Độ chính xác của model: {}".format(model.score(X_test, Y_test)))

    # Dùng tệp test để kiểm tra kết quả dựa trên model vừa huấn luyện
    print(model.predict(X_test))
    print(Y_test)

    # biểu đồ thể hiện sự phân tán dữ liệu giữa 2 cột Mileage và Price
    # plt.scatter(df['Mileage'], df['Sell Price($)'])

    # biểu đồ thể hiện sự phân tán dữ liệu giữa 2 cột Age (yrs) và Price
    # plt.scatter(df['Age(yrs)'], df['Sell Price($)'])
    plt.show()