
from __future__ import print_function
import sys
import numpy as np
import math
from operator import add

from random import randint

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

from pyspark import SparkContext, SparkConf

from pyspark.sql import SQLContext

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder



class ExplicitRating_engine(object):
    """A recommendation engine
    """
    def __init__(self, sc, rank, seed, iterations, regularization):
        """Init the recommendation engine given a Spark context and a dataset path
        """

        rawRatings = sc.textFile(sys.argv[1], 1)

        self.sc = sc

        ''' --- Load ratings data for later use ----'''
        #rawRatings = self.sc.textFile('data/test_ratings.csv')

        self.ratings_RDD =rawRatings \
            .map(lambda line: line.split(",")).map(lambda x: Rating(int(x[0]),int(x[1]),float(x[2]))).cache()
        #print (self.ratings_RDD.take(2))
        print ('number of ratings: ', self.ratings_RDD.count())
        print ('number of users: ', self.ratings_RDD.map(lambda x: (x.user, 1)).reduceByKey(add).count())
        print ('number of beers: ', self.ratings_RDD.map(lambda x: (x.product, 1)).reduceByKey(add).count())
        
        ''' ----- Load movies data for later use ----'''
        #rawMovies = self.sc.textFile('.csv')
        #print rawMovies.take(2)
        #self.movies_RDD = rawMovies \
        #    .map(lambda line: line.split(",")).map(lambda x: (int(x[0]),x[1])).cache()
        #print self.movies_RDD.take(2)
        #print 'number of beers = ', self.movies_RDD.count()
        
        
        #self.count_and_average_ratings()  
        
        '''  separate the data to training, validation and test sets'''
        weights = [.8, .2]
        seed = 42
        # Use randomSplit with weights and seed
        #self.trainData, self.valData, self.testData = self.ratings_RDD.randomSplit(weights, seed)
        self.trainData, self.testData = self.ratings_RDD.randomSplit(weights, seed)

        '''baseline model performance for testset'''
        self.estimate_baseline_model(self.testData)
        
        #print 'ok?'
        ''' Train the model and default parameters'''
        self.rank = rank
        self.seed = seed
        self.iterations = iterations
        self.regularization = regularization
        
        self.model_grid_search()
        print ('grid search is done')
        self.train_model()

# -------------------------------------------------------------------


    def count_and_average_ratings(self):
        """Updates the movies ratings counts from the current data self.ratings_RDD"""
        movie_ID_counts = dict(self.ratings_RDD.map(lambda x: (x[1], 1)).reduceByKey(add).collect())
        """note: self.ratings_RDD doesn't have complete movie list since not all movies are rated."""
        for movie, title in self.movies_RDD.collect():
                if movie not in movie_ID_counts: movie_ID_counts[movie] = 0

        self.movie_ID_avgRating_RDD = self.ratings_RDD.map(lambda x: (x[1], x[2]/movie_ID_counts[x[1]])).\
                    reduceByKey(add)
        
        self.movie_ID_counts = movie_ID_counts
 
# ------------------------------------------------------------------------

    def estimate_baseline_model(self, user_movie_RDD):
        movie_counts = dict(self.ratings_RDD.map(lambda x: (x[1], 1)).reduceByKey(add).collect())
        movie_avgRating = dict(self.ratings_RDD.map(lambda x: (x[1], x[2]/movie_counts[x[1]])).reduceByKey(add).collect())

        #print(user_movie_RDD.take(5))
        #print(movie_avgRating.take(5))        
        print ('baseline model: ', math.sqrt(user_movie_RDD.map(lambda x: (x.rating - movie_avgRating[x.product])**2).mean()))
        print ('-------------------')

# ------------------------------------------------------------------------

    def train_model(self):
        """Train the ALS model with the current dataset using one hyperparemeters
        """
        print (self.rank, self.iterations, self.regularization)
        self.model = ALS.train(self.trainData, rank = self.rank, seed = self.seed, \
                               iterations=self.iterations, lambda_=self.regularization)
        self.ratings_accuracy(self.testData)
        
#  --------------------------------------------------------------------------
        
    def model_grid_search(self):
        '''the grid search on parameters for ALS model'''
        
        num_cvFold = 10   ## number of cross-validation

        ranks = [5, 10, 15, 20, 40, 80, 120, 160]
        RMSEs = []
        MAEs = []
        min_error = float('inf')
        best_rank = -1
        best_reg = -1.0
        best_iteration = -1
        best_model = None

        '''here we want to do grid search'''
        for rank in ranks:
            for reg in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 2.0]:

                RMSE =0.0
                MAE =0.0
                for cv in range(num_cvFold):   ### using cross-validation 
                    weights = [.8, .2]
                    train, val = self.trainData.randomSplit(weights, seed=randint(0,100))

                    model = ALS.train(train,rank,seed=self.seed,iterations=self.iterations,lambda_=reg)

                    X_val = val.map(lambda x: (x[0], x[1]))
                    predictions = model.predictAll(X_val).map(lambda r: ((r[0], r[1]), r[2]))
                    rates_and_preds = val.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
                    RMSE = RMSE+math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
                    MAE = MAE+rates_and_preds.map(lambda r: abs(r[1][0] - r[1][1])).mean()

                    
                RMSE= RMSE/num_cvFold
                MAE = MAE/num_cvFold
                RMSEs.append(RMSE)
                MAEs.append(MAE)
                print ('For rank %s, reg %s, the RMSE is %s, MAE is %s' % (rank, reg, RMSE, MAE))
                if RMSE < min_error:
                    min_error = RMSE
                    best_model = model
                    best_rank = rank
                    best_reg = reg
        '''the grid search is done'''
        print ('The best model was trained with rank %s and reg %s' % (best_rank, best_reg))
        print ('The validation accuracy is', min_error)
        self.rank = best_rank
        self.regularization = best_reg
        self.model = best_model
        
# ----------------------------------------------------------------
    
    def ratings_accuracy(self, test_ratings_RDD):
        """Gets predictions for a given (userID, movieID) formatted RDD
        this is for checking the prediction accuracy
        """
        predicted_ratings_RDD = self.get_predict_ratings(test_ratings_RDD)
        ratings_and_preds = test_ratings_RDD.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).\
                join(predicted_ratings_RDD)
        error = math.sqrt(ratings_and_preds.map(lambda x: (x[1][0] - x[1][1])**2).mean())
        print ('the RMSE for the model is %s' %  error)

# --------------------------------------------------------
    
    def get_predict_ratings(self, user_movie_RDD):
        """Gets predictions for a given (userID, movieID) formatted RDD
        this is for checking the prediction accuracy
        """
        newMovie_rating = user_movie_RDD.map(lambda x: (x[0], x[1]))
        predicted_rating_RDD = self.model.predictAll(newMovie_rating).map(lambda x: ((x[0], x[1]), x[2]))
        return predicted_rating_RDD

# ---------------------------------------------------------

    def recommend_top_movies(self, user_movies_RDD):
        """Recommends up to movies_count top unrated movies to user_id
        """
        print (user_movies_RDD.take(10))
        # Get pairs of (userID, movieID) for user_id unrated movies
        
        the_user = user_movies_RDD.map(lambda x: x[0]).take(1)[0]
        print (the_user)
        
        user_rated_movies_ids = user_movies_RDD.map(lambda x: x[1]).collect() # get rated movie IDs, a list
        
        #print user_rated_movies_ids
        
        user_unrated_movies_RDD = self.movies_RDD.filter(lambda x: x[0] not in user_rated_movies_ids).\
                    map(lambda x: (the_user, x[0]))
        
        #print user_unrated_movies_RDD.take(10)
        # Get predicted ratings
        print (self.get_predict_ratings(user_unrated_movies_RDD).take(5))
        
        """note in the lambda function module is not applicable, i.e. (lambda x: self..()) doesn' work!"""
        movie_ID_counts = self.movie_ID_counts
        
        ratings = self.get_predict_ratings(user_unrated_movies_RDD).\
              filter(lambda x: movie_ID_counts[x[0][1]] > 1).sortBy(lambda x: -x[1])
        
        #print ratings.take(100)
        
        movie_titles = dict(self.movies_RDD.collect()) 
        '''note self. module is not allowed to use in lambda function'''
        #print movie_titles
        
        sortedRatings_recommd = ratings.map(lambda x: (x[0][1], movie_titles[x[0][1]], movie_ID_counts[x[0][1]], x[1]))
        
        sqlContext = SQLContext(self.sc)
        schema = sqlContext.createDataFrame(sortedRatings_recommd)
        schema.registerTempTable("recommend_table")
        pd = sqlContext.sql("SELECT * from recommend_table limit 50")
        pd.show()
 



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="beers_beHoppy_ratings")
    engine = ExplicitRating_engine(sc, rank=8, seed=32, iterations=20, regularization=0.06)
