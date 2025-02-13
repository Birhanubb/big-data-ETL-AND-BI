# Import all libraries
import pandas as pd
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import duckdb

# Data Extraction
file_path = 'C:/Users/POST LAB/big_data/data1.csv'  # The file path to the data

# Load the CSV file
data = pd.read_csv(file_path)

# Display the first 5 rows to get an overview of the dataset
print("Initial data preview (using Pandas):")
print(data.head())

# Get the shape of the dataset to see its dimensions (rows and columns)
print(f"\nDataset Shape: {data.shape}")

# Checking for missing values in each column
missing_values = data.isnull().sum()
print("\nMissing values in the dataset (Pandas):")
print(missing_values)

# Data Transformation using PySpark
# Initialize PySpark
spark = SparkSession.builder \
    .appName("EcommerceDataTransformation") \
    .getOrCreate()

# Load the data into a PySpark DataFrame
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Display the first 5 rows of the Spark DataFrame
print("\nFirst 5 rows of Spark DataFrame:")
df_spark.show(5)

# Transformation 1: Clean 'CustomerDOB' column by filling missing values with a default date
df_spark_cleaned = df_spark.fillna({"CustomerDOB": "01/01/1900"})

# Transformation 2: Handle missing 'CustAccountBalance' by filling it with 0
df_spark_cleaned = df_spark_cleaned.fillna({"CustAccountBalance": 0.0})

# Transformation 3: Remove rows where the 'price' is less than or equal to 0 (invalid entries)
df_spark_cleaned = df_spark_cleaned.filter(df_spark_cleaned["price"] > 0)

# Transformation 4: Convert 'TransactionDate' column to proper Date type
df_spark_cleaned = df_spark_cleaned.withColumn("TransactionDate", df_spark_cleaned["TransactionDate"].cast("date"))

# Transformation 5: Remove duplicate entries based on 'TransactionID'
df_spark_cleaned = df_spark_cleaned.dropDuplicates(subset=["TransactionID"])

# Convert PySpark DataFrame to Pandas DataFrame for further analysis
df_final_pandas = df_spark_cleaned.toPandas()

# Using DuckDB for aggregation
# Connect to DuckDB (in-memory database for fast aggregation)
con = duckdb.connect(database=':memory:', read_only=False)

# Register the DataFrame as a table in DuckDB
con.execute("CREATE TABLE ecommerce_sales AS SELECT * FROM df_final_pandas")

# Perform an aggregation: total purchases per customer
query = """
    SELECT
        "CustomerID",
        SUM("price") AS total_sales
    FROM ecommerce_sales
    GROUP BY "CustomerID"
    ORDER BY total_sales DESC
    LIMIT 50  -- Limit to top 50 customers by total sales to avoid overwhelming the plot
"""

# Execute the query and fetch the result
aggregated_data = con.execute(query).fetchall()

# Convert result to a DataFrame
aggregated_df = pd.DataFrame(aggregated_data, columns=["CustomerID", "total_sales"])

# Display the aggregated data (top 50 customers)
print("\nAggregated Data (Top 50 Customers by Total Purchases):")
print(aggregated_df.head())

# Visualization (Optimized)

# 1. Bar chart for the top customers by total sales
plt.figure(figsize=(10, 6))
plt.bar(aggregated_df['CustomerID'], aggregated_df['total_sales'], color='skyblue')
plt.xlabel('Customer ID')
plt.ylabel('Total Purchases ($)')
plt.title('Top 50 Customers by Total Purchases')
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.show()

# 2. Scatterplot: Visualizing the relationship between customer ID and total sales
plt.figure(figsize=(10, 6))
plt.scatter(aggregated_df['CustomerID'], aggregated_df['total_sales'], color='skyblue')
plt.xlabel('Customer ID')
plt.ylabel('Total Purchases ($)')
plt.title('Scatterplot: Total Purchases vs Customer ID')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# 3. Line Chart: A line chart to show sales trends for top customers
plt.figure(figsize=(10, 6))
plt.plot(aggregated_df['CustomerID'], aggregated_df['total_sales'], marker='o', color='skyblue', linestyle='-', markersize=5)
plt.xlabel('Customer ID')
plt.ylabel('Total Purchases ($)')
plt.title('Sales Trends for Top 50 Customers')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

# 4. Donut Chart: A donut chart for the total sales distribution across customers
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(aggregated_df['total_sales'], labels=aggregated_df['CustomerID'], autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
# Create the donut effect by drawing a white circle in the center
centre_circle = plt.Circle((0, 0), 0.50, color='white', fc='white', linewidth=1)
fig.gca().add_artist(centre_circle)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Donut Chart: Total Sales Distribution by Customer')
plt.show()

# Close the DuckDB connection
con.close()
