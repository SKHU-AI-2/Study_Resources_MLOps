from argparse import ArgumentParser
from sklearn.datasets import load_iris
import time
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

# 테이블 생성 함수
def create_table(db_connect):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS iris_data (
        id SERIAL PRIMARY KEY,
        timestamp timestamp,
        sepal_length float8,
        sepal_width float8,
        petal_length float8,
        petal_width float8,
        target int
    );"""
    print(create_table_query)
    # 작성한 query 를 DB 에 전달
    with db_connect.cursor() as cur:
        cur.execute(create_table_query)
        db_connect.commit()

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
        );
    """
    print(insert_row_query)
    # 작성한 query 를 DB 에 전달
    with db_connect.cursor() as cur:
        cur.execute(insert_row_query)
        db_connect.commit()

# Loop 추가
def generate_data(db_connect, df):
    while True:
        insert_data(db_connect, df.sample(1).squeeze()) # iris 데이터 한 줄 삽입
        time.sleep(1) # 데이터를 삽입 후 잠시 대기하는 시간(1초)

# DB 연결
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--db-host", dest="db_host", type=str, default="localhost")
    args = parser.parse_args()
    db_connect = psycopg2.connect(
        user="myuser",
        password="mypassword",
        host=args.db_host,
        port=5432,
        database="mydatabase",
    )
    create_table(db_connect)
    df = get_data()
    generate_data(db_connect, df)