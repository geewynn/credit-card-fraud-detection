import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef

st.set_option('deprecation.showfileUploaderEncoding', False)



def compute_performance(model, X_train, y_train,X_test,y_test):
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
    'Accuracy: ',round(scores,3)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm=confusion_matrix(y_test,y_pred)
    'Confusion Matrix: ',cm  
    cr=classification_report(y_test, y_pred)
    'Classification Report: ',cr
    


# file upload section
file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
if file_upload is not None:
    data = pd.read_csv(file_upload)
    
    #st.sidebar.checkbox('Show what the dataframe looks like')
    st.write(data.head(5))
    st.write('Shape of the dataframe: ',data.shape)
    st.write('Data decription: \n',data.describe())

    #set feature and target data
    X=data.drop(['Class'], axis=1)
    y=data.Class

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    st.write('X_train: ',X_train.shape, ' y_train: ',y_train.shape)
    st.write('X_test: ',X_test.shape, ' y_test: ',y_test.shape)

    # Instantiate random forest model
    rforest=RandomForestClassifier(random_state=42)


    features=X_train.columns.tolist()


    #Feature selection through feature importance
    @st.cache
    def feature_sort(model,X_train,y_train):
        #feature selection
        mod=model
        # fit the model
        mod.fit(X_train, y_train)
        # get importance
        imp = mod.feature_importances_
        return imp

    # Get feature importance and plot it
    model=rforest
    importance=feature_sort(model,X_train,y_train)
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature (Variable Number)')
    plt.ylabel('Importance')
    st.pyplot()

    # get top features from the feature importance list
    feature_imp=list(zip(features,importance))
    feature_sort=sorted(feature_imp, key = lambda x: x[1])
    n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)
    top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

    if st.sidebar.checkbox('Show selected top features'):
        st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

    # get train and test data
    X_train_sfs=X_train[top_features]
    X_test_sfs=X_test[top_features]

    X_train_sfs_scaled=X_train_sfs
    X_test_sfs_scaled=X_test_sfs


    # set random seed
    np.random.seed(42)

    # set smote to handle imbalance class
    smt = SMOTE()

    # set random forest model
    model=rforest
    
    # print the shape of random forest model
    rect=smt
    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
    X_train_bal, y_train_bal = rect.fit_sample(X_train_sfs_scaled, y_train)
    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

