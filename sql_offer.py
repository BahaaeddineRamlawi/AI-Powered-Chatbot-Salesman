import sqlite3
import pandas as pd
import json

# Load CSV data
csv_file = "offers.csv"  # Update with the actual filename
df = pd.read_csv(csv_file)

# Connect to SQLite database (creates a new one if not exists)
conn = sqlite3.connect("offers_database.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS offers (
        id INTEGER PRIMARY KEY,
        title TEXT,
        price REAL,
        link TEXT,
        categories TEXT,
        description TEXT,
        rating REAL,
        weight TEXT,
        image TEXT,
        stock_status TEXT,
        product_list TEXT
    )
""")

# Convert product_list to JSON format before inserting
df["product_list"] = df["product_list"].apply(lambda x: json.dumps(x.strip("[]").split(",")))


# Insert data into the table
df.to_sql("offers", conn, if_exists="replace", index=False)

# Commit and close connection
conn.commit()



cursor = conn.cursor()

# Query to select all rows from the "offers" table
cursor.execute("SELECT * FROM offers")

# Fetch all results
rows = cursor.fetchall()

# Print the results
for row in rows:
    print(row)


conn.close()

# print("Data inserted successfully into SQLite database.")