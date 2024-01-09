from sklearn.datasets import load_iris
import pandas as pd
import psycopg2

# iris 데이터 불러오기
def get_data():
    X, y = load_iris(return_X_y=True, as_frame=True)
    df = pd.concat([X, y], axis="columns")
    rename_rule = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
    }
    df = df.rename(columns=rename_rule)
    return df

# 데이터 삽입 함수
def insert_data(db_connect, data):
    insert_row_query = f"""
    INSERT INTO iris_data
        (timestamp, sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (
            NOW(),
            {data.sepal_length},
            {data.sepal_width},
            {data.petal_length},
            {data.petal_width},
            {data.target}
        );"""
    print(insert_row_query)
    # 작성한 query 를 DB 에 전달
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()

# DB 연결
if __name__ == "__main__":
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host="localhost",
        port=5432,
        database="mydatabase",
    )
    df = get_data()
    insert_data(db_connect, df.sample(1).squeeze()) # iris 데이터 한 줄 삽입