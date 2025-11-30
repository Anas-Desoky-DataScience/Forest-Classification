import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from google.colab import drive

drive.mount('/content/drive')
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data for projects/covtype.csv')
df.sample(10)
df.info()
df.describe()

X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_df = pd.DataFrame(X_train_scaled,
                          columns=X_train.columns,
                          index=X_train.index)
X_test_df = pd.DataFrame(X_test_scaled,
                         columns=X_test.columns,
                         index=X_test.index)

n_train = min(10000, len(X_train_df))
n_test = min(10000, len(X_test_df))

X_train_sample = X_train_df.sample(n=n_train, random_state=42)
X_test_sample  = X_test_df.sample(n=n_test, random_state=42)

y_train_sample = y_train.loc[X_train_sample.index]
y_test_sample  = y_test.loc[X_test_sample.index]

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_sample, y_train_sample)
y_pred=rf.predict(X_test_sample)

print(classification_report(y_test_sample, y_pred))

cm=ConfusionMatrixDisplay(confusion_matrix(y_test_sample,y_pred))
cm.plot()
