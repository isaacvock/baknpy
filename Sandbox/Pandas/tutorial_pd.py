'''
This is some sandbox code for me to play with to learn Pandas
'''

import pandas as pd


### Creating a DataFrame or Series

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth"
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"]

    }
)

ages = pd.Series([22, 35, 58], name="Age")


### Operations on DataFrames and Series

# Find maximum of a DataFrame column
df["Age"].max()

# Find maximum of a Series
ages.max()

# Get a basic description of the DataFrame
df.describe()


# Summarization

data = {
    'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
    'B': ['one', 'one', 'two', 'two', 'one', 'one'],
    'C': [1, 2, 3, 4, 5, 6],
    'D': [10, 20, 30, 40, 50, 60]
}

df = pd.DataFrame(data)
grouped = df.groupby(['A', 'B']).agg({'C': 'sum', 'D': 'sum'}).reset_index()

grouped