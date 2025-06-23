import pandas as pd
import numpy as np

def make_training_set(transactions_path="../data/processed/clean_data.csv",
                      customer_features_path="../data/processed/customer_features.csv",
                      output_path="../data/processed/training_data.csv"):
    
    # load data
    df = pd.read_csv(transactions_path)
    customer_df = pd.read_csv(customer_features_path)

    # verify that the date format is correct
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # create a table of purchases by customer and month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    purchases = df.groupby(["CustomerID", "YearMonth"]).agg({
        "TotalPrice": "sum"
    }).reset_index()
    purchases["Label"] = 1  # Indicates that there was a purchase

    # Create a combination of all customers and all possible months
    all_months = df["YearMonth"].drop_duplicates().sort_values()
    all_customers = df["CustomerID"].drop_duplicates()
    customer_months = pd.MultiIndex.from_product(
        [all_customers, all_months], names=["CustomerID", "YearMonth"]
    ).to_frame(index=False)

    # merge with purchases to mark who didnt buy
    full_data = pd.merge(customer_months, purchases, how="left", on=["CustomerID", "YearMonth"])
    full_data["Label"] = full_data["Label"].fillna(0)  # איפה שאין רכישה – שים 0
    full_data["TotalPrice"] = full_data["TotalPrice"].fillna(0)

    # mege with purchases to mark who didn't buy
    final_df = pd.merge(full_data, customer_df, on="CustomerID", how="left")

    # convert back to date
    final_df["YearMonth"] = final_df["YearMonth"].astype(str)

    # saving
    final_df.to_csv(output_path, index=False)
    print(f" traning set saved to: {output_path}")
    return final_df


# test 
if __name__ == "__main__":
    make_training_set()
