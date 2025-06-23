import pandas as pd

def load_and_clean_data():
    # geting the raw data from data/raw
    df = pd.read_excel(os.path.join(BASE_DIR, "..", "data", "raw", "Online_Retail.xlsx"))

    # removing columns with missing values     
    df = df.dropna(subset=["CustomerID", "InvoiceNo", "Description", "InvoiceDate", "Quantity", "UnitPrice"])

    # convert to a date column
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # create a total price column
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # adding time features
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Hour"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek

    # save the file to: data/processed
    df.to_csv("../data/processed/clean_data.csv", index=False)

    print("The file saved to: data/processed/clean_data.csv")
    return df


# test
if __name__ == "__main__":
    load_and_clean_data()
