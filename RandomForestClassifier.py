import numpy as np
from math import sqrt
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample 

class RandomForestCustom :
    def __init__(self, n_estimators=100, random_state=None):
        # hyperparametres 
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # liste pour stocker la foret 
        self.trees = []


    def fit(self,X,y):
        self.trees = [] ; 
        for i in range (self.n_estimators):
            #le bootstraping 
            
            X_sample , y_sample = resample ( X , y ,
                                            replace = True,
                                            n_samples = len(X),
                                            random_state= i # comme ca chaque tree sera différent 
                                            )
            
            #creation de l'arbre avec du random subspace 

            n_features = int( sqrt( X.shape[1] ) )
            tree = DecisionTreeClassifier ( max_features= n_features , random_state=i)
            tree.fit(X_sample , y_sample)
            self.trees.append(tree)
        return self 


    def predict(self,X):  
        # mettre chaque participation d'arbre 
        preds = []  
        for tree in self.trees : 
            preds.append ( tree.predict(X) )

        preds = np.array(preds)
        # final preds sera juste la prediction qui est les plus sortie dans preds 
        final_preds = []
        for i in range(preds.shape[1]):
            individual_participation = preds[:, i]
            answer = np.bincount(individual_participation).argmax()
            final_preds.append(answer)
            
        return np.array(final_preds)
        