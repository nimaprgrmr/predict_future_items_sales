import numpy as np
import pandas as pd
import warnings
import pickle
import plotly.express as px
from datetime import date, timedelta
import re

warnings.filterwarnings("ignore")


# FOR TRAINING DATA
def read_data(path="all_sales.csv"):
    df = pd.read_csv(path, header=None)
    columns = ['id_factor', 'date', 'id_br', 'amount', 'id_gds', 'dsc_gds', 'gds_class', 'unit_price', 'id_client']
    df.columns = columns
    df = df.drop(['unit_price', 'id_client'], axis=1)
    df['gds_class'] = df['gds_class'].astype(str)
    return df


def make_new_date(column):
    year = str(column.split('-')[0])
    month = str(column.split('-')[1])
    day = str(column.split('-')[2][0:2])
    date = year + '-' + month + '-' + day
    return date


def make_count(column, word):
    freq_word = column.count(word)
    return freq_word


def make_new_df(df):
    df['new_date'] = df['date'].apply(make_new_date)
    # Create data for 51338 branch (Bamland)
    df_branch = df[df['id_br'] == 51338]
    df_branch = df_branch.groupby('new_date').sum()
    df_branch['date'] = df_branch.index
    df_branch['year'] = df_branch['date'].apply(lambda x: x.split('-')[0])
    df_branch['month'] = df_branch['date'].apply(lambda x: x.split('-')[1])
    df_branch['day'] = df_branch['date'].apply(lambda x: x.split('-')[2])
    df_branch['id_br'] = 51338
    df_branch['اورآل'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'اورآل'))
    df_branch['بادی'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بادی'))
    df_branch['بارانی'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بارانی'))
    df_branch['بافت'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بافت'))
    df_branch['بلوز'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بلوز'))
    df_branch['بلوز و شلوار کودک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بلوز و شلوار کودک'))
    df_branch['بلوز کودک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'بلوز کودک'))
    df_branch['تاپ'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'تاپ'))
    df_branch['تونیک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'تونیک'))
    df_branch['تی شرت'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'تی شرت'))
    df_branch['دامن'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'دامن'))
    df_branch['سارافون'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'سارافون'))
    df_branch['سایر'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'سایر'))
    df_branch['ست بلوز و شلوار'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'ست بلوز و شلوار'))
    df_branch['سویی شرت'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'سویی شرت'))
    df_branch['شال و روسری'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'شال و روسری'))
    df_branch['شلوار'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'شلوار'))
    df_branch['شلوار کودک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'شلوار کودک'))
    df_branch['شومیز'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'شومیز'))
    df_branch['ماسک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'ماسک'))
    df_branch['مانتو'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'مانتو'))
    df_branch['هودی کودک'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'هودی کودک'))
    df_branch['پارچه'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'پارچه'))
    df_branch['پالتو'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'پالتو'))
    df_branch['پیراهن'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'پیراهن'))
    df_branch['ژاکت'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'ژاکت'))
    df_branch['کاپشن'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کاپشن'))
    df_branch['کت'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کت'))
    df_branch['کت شلوار'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کت شلوار'))
    df_branch['کفش و صندل'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کفش و صندل'))
    df_branch['کلاه،هدبند،پاپوش'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کلاه،هدبند،پاپوش'))
    df_branch['کیف'] = df_branch['dsc_gds'].apply(lambda x: make_count(x, 'کیف'))
    df_branch = df_branch.drop(['id_factor', 'amount', 'dsc_gds', 'id_gds', 'gds_class'], axis=1)
    return df_branch


def train_model(train_data):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score
    import copy

    new_data = copy.deepcopy(train_data[:-45])
    scaler = StandardScaler()
    X = new_data[['year', 'month', 'day']]
    X_scaled = scaler.fit_transform(X)
    y = new_data.drop(['id_br', 'date', 'year', 'month', 'day'], axis=1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=1e-6, random_state=55)
    X_valid = train_data[-45:-1][['year', 'month', 'day']]
    X_valid = scaler.transform(X_valid)
    y_valid = train_data[-45:-1].drop(['id_br', 'date', 'year', 'month', 'day'], axis=1)

    np.random.seed(42)
    # model_knr = KNeighborsRegressor(n_neighbors=9, weights='uniform', algorithm='auto', p=0.8)
    model_rfr = RandomForestRegressor(n_estimators=2000, criterion='absolute_error')
    model_rfr.fit(X_train, y_train)

    # make predictions
    # preds = model_rfr.predict(X_valid).astype(int)


    # Calculate Mean Absolute Error (MAE)
    # mae = np.abs(y_valid - preds).mean()

    return model_rfr, scaler


df = read_data()
df = make_new_df(df)

# model, scaler = train_model(df)
# save the model to disk
# filename_model = 'rfr_model.pkl'
# file_name_scaler = 'scaler.pk'
# l1 = ['اورآل', 'بادی', 'بارانی', 'بافت', 'بلوز', 'بلوز و شلوار کودک', 'بلوز کودک', 'تاپ',
#       'تونیک', 'تی شرت', 'دامن', 'سارافون', 'سایر', 'ست بلوز و شلوار', 'سویی شرت', 'شال و روسری',
#      'شلوار', 'شلوار کودک', 'شومیز', 'ماسک', 'مانتو', 'هودی کودک', 'پارچه', 'پالتو', 'پیراهن', 'ژاکت', 'کاپشن',
#       'کت', 'کت شلوار', 'کفش و صندل', 'کلاه، هدبند، پاپوش', 'کیف']
# pickle.dump(model, open(filename_model, 'wb'))
# pickle.dump(scaler, open(file_name_scaler, 'wb'))

# load the model from disk
# loaded_model = pickle.load(open(filename_model, 'rb'))



# for see the validation data
# valid = df.iloc[1557:1587] #
# valid = df.iloc[1451:1482] # 1 ordibehesht ta 1 khordad
# valid = df.iloc[1420:1451]  # 1 farvardin ta 1 ordibehesht
# print(valid)
# valid = df.iloc[1451:1482].drop(['date', 'id_br', 'year', 'month', 'day'], axis=1)
# total_valid = 0
# for i in valid.values:
#     total_valid += i
# print(total_valid)
