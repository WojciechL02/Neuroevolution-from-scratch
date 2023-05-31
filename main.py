import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def prepare_dataset():
    df = pd.read_csv("data/Wine_Quality_Data.csv")
    df["color"] = df["color"].map({"red": 1, "white": 0})
    df["quality"] = df["quality"] - 3
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def main():
    pass


if __name__ == '__main__':
    main()
