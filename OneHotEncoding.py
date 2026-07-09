import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
'''
C1 (phiên bản cũ): LabelEncoder trước (đưa cột base về số nguyên) 
-> OneHotEncoder đưa cột đó về 0/1 với từng category (do OneHotEncoder cũ không
xử lí string)
C2 (phiên bản mới): Dùng thằng OneHotEncoder vì đã hỗ trợ String
'''

if __name__ == '__main__':
    df = pd.read_csv("data/dummy_onehotencoding.csv")

    # # Tạo object mã hóa label
    # le = LabelEncoder()
    #
    # # Mã hóa data frame
    # df_le = df
    # df_le['town'] = le.fit_transform(df_le['town'])
    model = LinearRegression()

    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), [0])], # Mã hóa cột đầu tiên
        remainder='passthrough' # Dữ nguyên các cột còn lại
    )
    X_train = df[['town', 'area']].values
    Y_train = df['price']
    X = ct.fit_transform(X_train)

    # Loại bỏ 1 cột dummy để tránh dummy trap
    X = X[:, 1:]
    model.fit(X, Y_train)

    # Thử dự đoán
    print(model.predict([[0, 1, 3400]]))
