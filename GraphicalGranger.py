from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import grangercausalitytests

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
import numpy as np
import pylops

#X, y_vals = generate_data_lasso_problem(x,y, 5)
#print(X.shape, y_vals.shape)
#xinv = calculate_fista_model(X, y_vals, 0.01)
#calculate_error(xinv, X,y_vals)

class FistaEstimator(BaseEstimator):
   def __init__(self, alpha=0.01):
      self.alpha = alpha
      
   def calculate_fista_model(self, X, y, eps):
    k = np.sqrt(1/(2*len(y)))
    X = k*X
    y = k*y

    Aop = pylops.MatrixMult(X)
    maxit = 1000
    xinv, niter, cost = pylops.optimization.sparsity.FISTA(Aop, y, maxit, eps=eps, tol=1e-5, returninfo=True)
    return xinv, cost


   def calculate_error(self, x, X, y):
      #err = np.linalg.norm(y.T - np.matmul(X,x.T)) + eps * np.linalg.norm(x, ord=1)
      #err = (1 / (2 * len(y))) * np.linalg.norm(y.T - np.matmul(X,x.T))**2 + eps * np.linalg.norm(x, ord=1)
      predicted_val = np.matmul(X,x.T)
      err = (1 / (2 * len(y))) * np.linalg.norm(y.T - np.matmul(X,x.T))**2 + eps * np.linalg.norm(x, ord=1)
      err = -np.linalg.norm(y.T - np.matmul(X,x.T))**2 /len(y)
      return err
    
   def fit(self, X, y):
      self.coef_, cost = self.calculate_fista_model(X, y, self.alpha)
      #self.score_ = cost[len(cost)-1]
      #print(cost)
      
   def score(self, X, y, sample_weight=None):
      y_pred = self.predict(X)
      return r2_score(y, y_pred, sample_weight=sample_weight)
      
   def predict(self, X):
      return np.matmul(X,self.coef_.T)

class GraphicalGranger:
    minimal_alpha = 0.2
    alpha = None
    time_series = []
    
    def __init__(self,alpha_min=0.2):
        self.minimal_alpha = alpha_min
        
    def graphical_relations_matrix(self, time_series_array, lag, fixed_alpha=None, use_model="lasso"):
        time_series_array = np.array(time_series_array)
        matrix = []
        for i in range(len(time_series_array)):
            x = time_series_array[i]
            y_array = time_series_array[~np.isin(np.arange(len(time_series_array)), [i])]
            connections = self.graphical_relations(x, y_array,lag, fixed_alpha, use_model)
            connections = connections[1:]
            connections = np.insert(connections, i, 1)
            
            print(connections)

            matrix.append(connections)
        return matrix

    def graphical_relations(self, x, y_array,lag, fixed_alpha=None, use_model="lasso"):
           if use_model == "lasso" or use_model == "fista":
               return self.graphical_relations_optimiz(x, y_array,lag, fixed_alpha, use_model)
           else: 
               return self.graphical_relations_classical(x,y_array, lag)
                
    def graphical_relations_classical(self, x, y_array,lag):
        conections_array = [1]
        for i in range(len(y_array)):
            connected = self.is_causal_classic(x,y_array[i], lag)
        
            if connected == True :
                conections_array.append(0)
            else:
                conections_array.append(1)
        return np.array(conections_array)
    
    def is_causal_classic (self, x, y,lag): 
        df = pd.DataFrame({'x': x, 'y':  y}) 
        return grangercausalitytests(df[['x', 'y']], maxlag=[lag])[lag][0]["ssr_ftest"][1] > 0.05
        
    def graphical_relations_optimiz(self, x, y_array,lag, fixed_alpha=None, use_model="lasso"):
        if use_model == "lasso":
            model = Lasso
        elif use_model == "fista":
            model = FistaEstimator

        data, data_x, x_vals = self.generate_data_lasso_problem_multiple(y_array, x, lag)

        if fixed_alpha is None:
            alpha = self.obtain_alpha_model_using_cross_validation(data,x_vals, model)
        else:
            alpha = fixed_alpha

        clf = model(alpha=alpha)
        res = clf.fit(data,x_vals)
        coefs = clf.coef_
        
        conections_array = []
        for i in range(len(y_array)+1):
            coefs_x_x_i = coefs[(lag-1)*i: (lag-1)*(i+1)]
            all_zeros = not np.any(coefs_x_x_i)
            if all_zeros == True :
                conections_array.append(0)
            else:
                conections_array.append(1)
        return np.array(conections_array)
    
    def generate_data_lasso_problem_multiple(self, x_array, y, L):
    
        data_x_array = []
        data_y = self.apply_lagg(y,L)
        
        data_x_array.append(data_y) 
        
        for x in x_array:
            data_x_array.append(self.apply_lagg(x,L))
        
        y_vals = []
        for t in range(L,len(x)):
            y_t = y[t-1]
            y_vals.append(y_t)
        y_vals.reverse()
            
        data = np.column_stack(data_x_array)
            
        return data, np.array(data_y), np.array(y_vals)
    
    def apply_lagg(self, x,L):
        data_x = []
        for t in range(L,len(x)):
            x_prev = x[t-L:t-1]
            data_x.append(x_prev)
        data_x.reverse()
        return np.array(data_x)
    
    def obtain_alpha_model_using_cross_validation(self, data, x_vals, model):
        
        try:
            grid = dict()
            grid['alpha'] = np.arange(self.minimal_alpha, 2, 0.01)
            
            cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)

            search = GridSearchCV(model(), grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=7)

            results = search.fit(data, x_vals)
            # summarize
            print('MAE: %.3f' % results.best_score_)
            print('Config: %s' % results.best_params_)

            alpha = results.best_params_["alpha"]
            return alpha
        except:
            print("error, returning minimal alpha")
            return self.minimal_alpha

    def create_graph(self, connections_matrix, time_series_labels=None):
        if time_series_labels == None:
            time_series_labels = list(range(len(connections_matrix)))
        
        G = nx.Graph()

        for i in range(len(connections_matrix)):
            for j in range(len(connections_matrix[i])):
                connected = connections_matrix[i][j] == 1
                if connected and not i == j: 
                    G.add_edge(time_series_labels[i], time_series_labels[j])
                
        return G
    
    def draw_graph(self, connections_matrix, time_series_labels=None):
        G = self.create_graph(connections_matrix, time_series_labels)
        options = {
            "font_size": 36,
            "node_size": 3000,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 5,
            "width": 5,
        }
        
        nx.draw_networkx(G,  **options)

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()