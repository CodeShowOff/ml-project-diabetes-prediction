import pandas as pd

data = {'name': ['Ashish', 'Raj', 'Ram'],
        'age': [27, 39, 22],
        'salary': [50000, 20200, 39423]
}


df = pd.DataFrame(data)
print(df)

print(df.describe())