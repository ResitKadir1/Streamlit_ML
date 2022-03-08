
#from typing_extensions import dataclass_transform
import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,plot_confusion_matrix, confusion_matrix
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
st.title("Classification Model Decision")
#Run --> streamlit run MultiClass_App.py 

st.write("""
##### Model Train and Testing

 By Reka
""")

dataset_name=st.sidebar.selectbox("Select Dataset",("Iris","Breast Cancer","Wine dataset"))
#st.write(dataset_name)
classifer_Name=st.sidebar.selectbox("Select Classifier",("KNN","SVM","Random Forest"))


def get_dataset(data):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    #Split test and train
    X =data.data
    y = data.target
    return X,y


X, y =get_dataset(dataset_name)


def add_parameter_ui(clf_name):
    params = dict()#open an empty dictionary

    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"]=K # {"K":1-15}
    elif clf_name == "SVM":
         C = st.sidebar.slider("C",0.1,10.0)
         params["C"]=C
    else:
        #clf_name == "Random Forest":
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("number of estimators",1,100)
        params["max_depth"]=max_depth
        params["n_estimators"]=n_estimators

    return params

params=add_parameter_ui(classifer_Name)

def get_classifier(clf_name,params):

    if clf_name == "KNN":
        clf =KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        #clf_name == "Random Forest":
        #max_depth = st.sidebar.slider("max_depth",2,15)
        #n_estimators = st.sidebar.slider("number of estimators",1,100)

        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth =params["max_depth"],random_state=1234)

    return clf
clf = get_classifier(classifer_Name,params)


##Classification
X_train,X_test ,y_train,y_test =train_test_split(X,y,test_size = 0.2,random_state=1234)

#Train model
clf.fit(X_train,y_train)

#y_test = y_test.reshape(1,-1)

#make predicition
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test,y_pred)
#st.write(f"Classifier={classifer_Name}")

st.write(f"Accuracy={acc}")

cf=confusion_matrix(y_test,y_pred)
#st.write(f"confusion matrix={cf}")


clf = get_classifier(classifer_Name,params)
X,y = get_dataset(dataset_name)#which defined column13
st.write("Shape of datasets",X.shape)
st.write("Number of the Classes",len(np.unique(y)))



#PLOOT

pca = PCA(2)#dimension
X_project =pca.fit_transform(X)

x1 =X_project[:,0]
x2 =X_project[:,1]
#x3 =X_project[:,2]
fig =plt.figure()
#fig.patch.set_facecolor('black')
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.colorbar()

st.set_option('deprecation.showPyplotGlobalUse', False)
#plt.show()
st.pyplot()