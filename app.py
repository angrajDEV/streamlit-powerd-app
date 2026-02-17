import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stApp {
    background-color: #DCBEE6;
}
[data-testid="stSidebar"] {
    background-color: #B199E8 !important;
    border-right: 1px solid #ddd8d0;
}
h1, h2, h3 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #2c3e50 !important;
}
.stButton > button {
    background-color: #5b8a72;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #4a7560;
    color: white;
}
[data-testid="stMetricValue"] {
    color: #2c3e50 !important;
}
[data-testid="stMetricLabel"] {
    color: #5b8a72 !important;
}
</style>
""", unsafe_allow_html=True)

st.title('ML-powered Streamlit App ')
st.subheader('coustemer churn prediction in different model\'s')
st.write('We have used Random Forest Feature Importance based Feature Selection method to select the most relevant features from the dataset. After feature selection, we compare different classification models to see which one has the most accurate prediction.')

file = st.file_uploader('upload the fie ', type=['csv'])

if file:
    df = pd.read_csv(file)
    st.subheader('data preview')
    st.dataframe(df)
    st.write(df.shape)

    if st.sidebar.button('take best features'):
        TARGET = "Churn"
        df_clean = df.copy()

        id_cols = [c for c in df_clean.columns if 'id' in c.lower()]
        df_clean.drop(columns=id_cols, inplace=True, errors='ignore')

        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

        if df_clean[TARGET].dtype == object:
            df_clean[TARGET] = df_clean[TARGET].map({'Yes': 1, 'No': 0})

        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

        le = LabelEncoder()
        cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
        cat_cols = [c for c in cat_cols if c != TARGET]
        for col in cat_cols:
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

        X = df_clean.drop(columns=[TARGET])
        y = df_clean[TARGET].astype(int)

        scorer = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        scorer.fit(X, y)

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': scorer.feature_importances_
        }).sort_values('Importance', ascending=False)

        final_features = importance_df[importance_df['Importance'] > 0.01]['Feature'].tolist()

        # session_state mein save karo
        st.session_state['X'] = X[final_features]
        st.session_state['y'] = y
        st.session_state['final_features'] = final_features
        st.session_state['importance_df'] = importance_df

    # feature results hamesha show karo — button ke bahar
    if 'final_features' in st.session_state:
        st.subheader('selected feature :')
        st.write(st.session_state['final_features'])
        st.subheader('features with there score :')
        st.write(st.session_state['importance_df'])

    selected_algo = st.sidebar.selectbox('select which classifier for prediction :', ['not selected', 'DecisionTreeClassifier', 'random forest', 'xgboost'])

    if selected_algo == 'DecisionTreeClassifier':
        if 'X' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']

            depth = st.sidebar.slider('max depth', 0, 30, 20, key='dt_depth')
            min_samples_split = st.sidebar.slider('min samples split', 2, 20, 10, key='dt_split')
            min_samples_leaf = st.sidebar.slider('min samples leaf', 1, 20, 5, key='dt_leaf')

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            st.write(accuracy_score(y_test, y_pred))
            st.write(classification_report(y_test, y_pred))

            matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.sidebar.warning('pehle "take best features" button dabao')

    if selected_algo == 'random forest':
        if 'X' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']

            depth = st.sidebar.slider('max depth', 0, 20, 5, key='rf_depth')
            n_estimators = st.sidebar.slider('n estimators', 50, 500, 200, key='rf_estimators')
            min_samples_split = st.sidebar.slider('min samples split', 2, 20, 10, key='rf_split')
            min_samples_leaf = st.sidebar.slider('min samples leaf', 1, 20, 5, key='rf_leaf')

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(max_depth=depth, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            st.write(accuracy_score(y_test, y_pred))
            st.write(classification_report(y_test, y_pred))

            matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.sidebar.warning('pehle "take best features" button dabao')

    if selected_algo == 'xgboost':
        if 'X' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']

            n_estimators = st.sidebar.slider('n estimators', 50, 500, 200, key='xgb_estimators')
            max_depth = st.sidebar.slider('max depth', 1, 20, 5, key='xgb_depth')
            learning_rate = st.sidebar.slider('learning rate', 0.01, 0.5, 0.05, key='xgb_lr')

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = XGBClassifier(eval_metric='logloss', random_state=42, max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            st.write(accuracy_score(y_test, y_pred))
            st.write(classification_report(y_test, y_pred))

            matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("Actual Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        else:
            st.sidebar.warning('pehle "take best features" button dabao')