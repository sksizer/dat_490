#Import Library
import pandas as pd

# Load dataset
# Import Data
path = "/content/sample_data/LLCP2023.csv"
dfn = pd.read_csv(path)

#Create Data Frame
df = dfn[['BPHIGH6','TOLDHI3','CVDINFR4',	'CVDCRHD4',	'CVDSTRK3',	'ASTHMA3',	'CHCSCNC1',	'CHCOCNC1','CHCCOPD3',	'ADDEPEV3','CHCKDNY2',	'HAVARTH4',	'DIABETE4',	'DEAF',	'BLIND','DIFFWALK',	'COVIDPO1',	'PREDIAB2',	'CNCRDIFF',	'CNCRTYP2',	'_CASTHM1',	 '_DRDXAR2', '_HLTHPL1']].copy()

# Convert Data Frame to .CSV for project
df.to_csv('LeanData.csv', index=False)
