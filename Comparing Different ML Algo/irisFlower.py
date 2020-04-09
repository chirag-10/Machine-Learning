# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import joblib
import pickle


class iris:
    
    def __init__(self):
        self.url ="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
        names=['s_length','s_width','p_length','p_width','class_']
        self.dataset = read_csv(self.url,names=names)
        print(self.dataset.shape)
        print(self.dataset.head(20))
        print(self.dataset.groupby('class_').size())

    def visualization(self):
        self.dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        pyplot.show()
        self.dataset.hist()
        pyplot.show()

    def sets(self):
        self.array = self.dataset.values
        X = self.array[:,0:4]
        y = self.array[:,4]
        self.Xtrain,self.Xval,self.ytrain,self.yval = train_test_split(X,y,test_size=0.2,random_state=1,shuffle=True)   #shuffle is by default True
        
    def training(self):   
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        #models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        #models.append(('CART', DecisionTreeClassifier()))
        #models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))  
        max=999
        self.results=[]
        self.names=[]

        for name,model in models:
            kFold = StratifiedKFold(n_splits = 10 , random_state=1 , shuffle=True)
            cv_results = cross_val_score(model,self.Xtrain,self.ytrain,cv=kFold,scoring='accuracy')
            self.results.append(cv_results)
            self.names.append(name)
            print('%s :  %f   (%f)' % (name,cv_results.mean(),cv_results.std()))
            if cv_results.mean()<max:
                final_model = model

        return final_model


    def comparealgo(self):
        pyplot.title('Algorithm Comparison')
        pyplot.boxplot(self.results,labels=self.names)
        pyplot.show()
        pyplot.hist(self.results)
        pyplot.show()
            
    def predict(self,model):
        model.fit(self.Xtrain,self.ytrain)
        predictions = model.predict(self.Xval)
        print(accuracy_score(self.yval, predictions))
        print(confusion_matrix(self.yval, predictions))
        print(classification_report(self.yval, predictions))   
        
    def saveData(self,model):
        filename = 'iris_pickled_data.pkl'
        joblib.dump(model,filename)

    def loadData(self):
        model_loaded = joblib.load('iris_pickled_data.pkl')
        predict(model_loaded)

obj = iris()
#obj.visualization()
obj.sets()
model = obj.training()

obj.comparealgo()
obj.predict(model)
obj.saveData(model)
obj.loadData()
