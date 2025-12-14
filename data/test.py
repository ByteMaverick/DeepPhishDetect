import pandas as pd
basic = pd.read_csv("basic_data.csv")
eng   = pd.read_csv("tier2_data.csv")

print((basic.url == eng.url).all())
