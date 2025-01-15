import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_data(csv_file_path, nrows=1000000):
    df = pd.read_csv(csv_file_path, nrows=nrows)

    # Drop unnecessary features
    drop_features = ["category_id", "price"]
    df = df.drop(drop_features, axis=1)
    df = df.replace([0, '0'], np.nan).dropna()

    # Adjust category code formatting
    df["category_code"] = df["category_code"].apply(lambda loc: str(loc).replace(".", " "))

    # Encode product, user, and session IDs
    enc = LabelEncoder()
    df["product_id"] = enc.fit_transform(df["product_id"])
    df["user_id"] = enc.fit_transform(df["user_id"])
    df["user_session"] = enc.fit_transform(df["user_session"])

    # Calculate event-related metrics
    is_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_view")
    is_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_cart")
    is_purchase = df[df["event_type"] == "purchase"].groupby(["user_id", "user_session", "product_id"]).size().reset_index(name="is_purchase")

    total_view = df[df["event_type"] == "view"].groupby(["user_id", "user_session"]).size().reset_index(name="total_view")
    total_cart = df[df["event_type"] == "cart"].groupby(["user_id", "user_session"]).size().reset_index(name="total_cart")
    total_purchase = df[df["event_type"] == "purchase"].groupby(["user_id", "user_session"]).size().reset_index(name="total_purchase")

    df = (df.merge(is_view, on=["user_id", "user_session", "product_id"], how="left")
            .merge(is_cart, on=["user_id", "user_session", "product_id"], how="left")
            .merge(is_purchase, on=["user_id", "user_session", "product_id"], how="left")
            .merge(total_view, on=["user_id", "user_session"], how="left")
            .merge(total_cart, on=["user_id", "user_session"], how="left")
            .merge(total_purchase, on=["user_id", "user_session"], how="left"))
    df.fillna(0, inplace=True)

    # Calculate duration and total duration
    df = df.sort_values(by=["user_id", "user_session", "event_time"], ascending=[True, True, True])
    df["event_time"] = pd.to_datetime(df["event_time"])
    df["duration"] = df.groupby(["user_id", "user_session"])["event_time"].diff().fillna(pd.Timedelta(seconds=0))
    df["duration"] = df["duration"].dt.total_seconds()
    df["total_duration"] = df.groupby(["user_id", "user_session"])["duration"].transform("sum")

    # Feature calculations
    df['F1'] = df.apply(lambda row: row['is_view'] / row['total_view'] if row['total_view'] != 0 else 0, axis=1)
    df['F2'] = df.apply(lambda row: row['is_cart'] / row['total_cart'] if row['total_cart'] != 0 else 0, axis=1)
    df['F3'] = df.apply(lambda row: row['is_purchase'] / row['total_purchase'] if row['total_purchase'] != 0 else 0, axis=1)
    df['F4'] = df.apply(lambda row: row['duration'] / row['total_duration'] if row['total_duration'] != 0 else 0, axis=1)

    # Calculate interest score
    initial_weights = {"F1": 0.1, "F2": 0.25, "F3": 0.45, "F4": 0.2}
    df["score"] = (initial_weights["F1"] * df["F1"] +
                   initial_weights["F2"] * df["F2"] +
                   initial_weights["F3"] * df["F3"] +
                   initial_weights["F4"] * df["F4"])
    df_weighted = df.groupby(["user_id", "product_id"])["score"].sum().reset_index(name="score")

    return df,df_weighted
