import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('./ds_large_keepout1_window25.csv')

train, test = train_test_split(data, train_size=0.8, random_state=40)

print(train.head(5), train.shape)

print(test.head(5), test.shape)

train.to_csv('./data/train.csv', index=False)
test.to_csv('./data/test.csv', index=False)
