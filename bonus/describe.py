import numpy as np
import pandas as pd
import sys
from scaler import Standard_Scaler
from TinyStatistician import TinyStatistician as TS

# disable false positive warnings
pd.options.mode.chained_assignment = None  # default='warn'


def describe_feature(x, df, feature):
    try:
        assert isinstance(df, pd.DataFrame) and isinstance(
            x, pd.DataFrame), "1st and 2nd arguments should be dataframe"
        assert feature in x.columns, "3rd argument is not a known feature"

        X_ = x[[feature]].values
        df.loc['Count', feature] = TS.countx(X_)
        df.loc['Mean', feature] = TS.meanx(X_)
        df.loc['Median', feature] = TS.median(X_)
        df.loc['Std', feature] = TS.std(X_)
        df.loc['Var', feature] = TS.var(X_)
        df.loc['Min', feature] = TS.minx(X_)
        df.loc['Max', feature] = TS.maxx(X_)
        df.loc['25%', feature] = TS.percentile(X_.reshape(-1,), 25)
        df.loc['50%', feature] = TS.percentile(X_.reshape(-1,), 50)
        df.loc['75%', feature] = TS.percentile(X_.reshape(-1,), 75)
        return df

    except Exception as e:
        print(e)
        return None


def fill_zeros(x):
    try:
        assert isinstance(
            x, pd.DataFrame), "argument should be a dataframe"
        for feature in labels:
            x[feature] = x[feature].fillna(0)
        return x

    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    try:
        assert len(sys.argv) >= 2, "missing path"
        path = sys.argv[1]

        # 1. Load data
        # path = "datasets/dataset_train.csv"
        data_train = pd.read_csv(path)

        # labels = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
        labels = data_train.select_dtypes(include=['int64', 'float64']).columns

        x_train = data_train[labels]

        # 2. fill empty cells with 0
        x_train = fill_zeros(x_train)

        # 3. create empty dataframe and fill it with describing parameters
        index = ['Count', 'Mean', 'Median', 'Std',
                 'Var', 'Min', '25%', '50%', '75%', 'Max']
        df = pd.DataFrame(columns=labels[1:], index=index)
        for feature in labels:
            df = describe_feature(x_train, df, feature)
        print(df)

    except FileNotFoundError:
        print("FileNotFoundError -- strerror: No such file or directory")

    except Exception as e:
        print(e)
