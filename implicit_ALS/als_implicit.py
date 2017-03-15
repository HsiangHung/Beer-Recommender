# Spark UI: http://ec2-54-86-54-194.compute-1.amazonaws.com:8080
# storage: http://ec2-54-86-54-194.compute-1.amazonaws.com:4040/storage
# slides: http://www.slideshare.net/erikbern/collaborative-filtering-at-spotify-16182818/59
## each node with 8GB is not enough. need more
from __future__ import print_function
import sys
import math
import numpy as np
from operator import add

#import pydoop.hdfs as hdfs
#import subprocess

from random import randint

from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating

from pyspark import SparkContext, SparkConf

from pyspark.sql import SQLContext

from ranking import Rank_list

# ---------------------------------------------------------------

def logConfidence(r):
    return Rating(r.user, r.product, math.log(1.0050+float(r.rating)*100000000)+float(r.rating))
    #return Rating(r.user, r.product, math.log(1.0050+float(r.rating))+float(r.rating))
    #return Rating(r.user, r.product, math.log(1.0050+float(r.rating)*100))

# --------------------------------------------------------------

def confidence(r):
    #if r.rating == 1:
    #    x = 1.0
    #elif r.rating > 1:
    #    x = 0.0
    if r.rating <= 2:
        x = r.rating
    elif r.rating > 2:
        x = 0.0
    return Rating(r.user, r.product, float(r.rating-x))

# --------------------------------------------------------------

#def like_or_unlike(r):
#    if float(r[1]) >0.70:
#        p = 1.0
#    else:
#        p = 0.0
#    return ((int(r[0][0]), int(r[0][1])), p)

# ----------------------------------------------------------------------------

def model_rank_score(r, rank_lists):
    ''' 
     input: r = (user, [(beer1, rating1, prob1), (beer2, rating2, prob2), ....]) 
     return:
              \sum_i rating_{u,i} * ranking_{u,i} for each user
    '''
    items = r[1]
    ## sort by probability in desc order; highest prob has early rank:
    items.sort(key=lambda tup: -tup[2])

    ## items[i][0] = i-th beer_id for user r[0]
    ## items[i][1] = i-th rating for user r[0]
    ## items[i][2] = probability for user r[0]

    length = len(items)
    rating_ranking =0.0
    #sum_ratings = 0.0
    for i in range(length):
        rating_ranking += items[i][1]*rank_lists[length][i]
        #sum_ratings += items[i][1]

    #return (rating_ranking, sum_ratings)
    return rating_ranking

# ---------------------------------------------------------------------------- 




class ImplicitCF_engine(object):
    """A collaborative filtering recommender engine using implicit feedback datasets:
     *spark.apache.org/docs/1.4.0/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.trainImplicit
     *spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.trainImplicit
     *spark.apache.org/docs/0.8.1/api/mllib/org/apache/spark/mllib/recommendation/ALS$.html
    """
    def __init__(self, sc, rank, seed, iterations, reg_parameter):
        """Init the recommendation engine given a Spark context and a dataset path"""
        #records = sc.textFile("hdfs://54.86.54.194:9000/test/records.csv")

        text = sc.textFile(sys.argv[1], 1)

        header = text.take(1)[0]
        
        text = text.filter(lambda x: x!= header)

        #print (text.take(5))

        self.buy_records_RDD =text.map(lambda x: x.split(",")).map(lambda x: (str(x[0]), str(x[1])))

        self.get_customers()

        self.n_beers, self.best_beers = self.get_beers()

        self.percent_rankings = Rank_list(self.n_beers).rank_list

        #print('count', self.buy_records_RDD.count())

        #print (self.best_beers)



        users_orders_RDD = self.buy_records_RDD.map(lambda x: ((x[0],x[1]), 1)).\
                         reduceByKey(lambda x, y: x+y).\
                         map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1])))

        users_orders_RDD = users_orders_RDD.map(confidence) ## make rating =1 => 0, otherwise won't change


        #   ----- combine the recent records ---------------
        text = sc.textFile("hdfs://54.86.54.194:9000/test/recent.csv", 1)
        header = text.take(1)[0]
        self.buy_records_RDD = text.filter(lambda x: x!= header).map(lambda x: x.split(",")).map(lambda x: (str(x[0]), str(x[1])))
        self.buy_records_RDD = self.buy_records_RDD.map(lambda x: ((x[0],x[1]), 1)).\
                         reduceByKey(lambda x, y: x+y).\
                         map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1])))

        #users_orders_RDD = users_orders_RDD.union(self.buy_records_RDD)
        users_orders_RDD = self.buy_records_RDD.map(confidence)


        #   ----- combine the more recent records ---------------
        text = sc.textFile("hdfs://54.86.54.194:9000/test/more_recent.csv", 1)
        header = text.take(1)[0]
        self.buy_records_RDD = text.filter(lambda x: x!= header).map(lambda x: x.split(",")).map(lambda x: (str(x[0]), str(x[1])))
        self.buy_records_RDD = self.buy_records_RDD.map(lambda x: ((x[0],x[1]), 1)).\
                         reduceByKey(lambda x, y: x+y).\
                         map(lambda x: Rating(int(x[0][0]), int(x[0][1]), float(x[1]))).\
                         map(lambda x: Rating(x.user, x.product, 2*x.rating))

        users_orders_RDD = users_orders_RDD.union(self.buy_records_RDD)



        print ('number of ratings: ', users_orders_RDD.count())


        #print (users_orders_RDD.collect())


        '''  separate the data to training, validation and test sets'''
        weights = [.6, .2, .2]
        # Use randomSplit with weights and seed
        self.trainData, self.valData, self.testData = users_orders_RDD.randomSplit(weights, randint(0,50))


        print ('n of val, test: ', self.valData.count(), self.testData.count())


        #self.testData = self.testData.filter(lambda x: x.rating > 10)
        #self.valData = self.valData.filter(lambda x: x.rating > 10)

        baseline_error = self.estimate_baseline_MPR()
        print (baseline_error)

        #return


        self.trainData = self.trainData.map(logConfidence)  ## more complicated confidence function


        #print ('ok?')

        ''' Train the model and default parameters'''
        self.rank = rank
        self.seed = seed
        self.iterations = iterations
        self.reg_parameter = reg_parameter

        #return        

        #self.train_model()            ### for single hyperparameter to run
        self.models_grid_search()      ### for mutli-hyperparameters to search
        print('model is done')
        self.estimate_MPR(self.testData)


#   ------------------------------------------------------------------------


    def get_beers(self):
        """
           Return:
                a list ranking best sold beers from historical transaction records:
                best_beers = [(br1, n1), (br2, n2), (br3, n3),.....], where n1>= n2 >= n3...
        """
        beers_counts = self.buy_records_RDD.map(lambda x: (x[1],1)).reduceByKey(add).collect()
        best_beers = sorted(beers_counts, key=lambda x: -x[1])
        n_beers = len(best_beers)
        print ('number of beers = ', n_beers)
        self.threshold_r = best_beers[100][1]
        print (best_beers[100])
        print('Top-10 rated beer:')
        for beer, purchases in best_beers[:10]:
            print(beer, purchases)

        return n_beers, best_beers


##   --------------------------------------------------------------------------


    def estimate_baseline_MPR(self):
        '''compute MPR, 'mean percentage ranking' for baseline model (average model)
           input:
                self.testData = (usr1, beer1, rating1), (usr2, beer2, rating2), ....
           return:
                sum_{u,i} r_{u,i} * rank_{u,i} / \sum_{u,i} r_{u,i}, where i = 1 ~ self.n_beers
           But in the baseline model, rank_{u,i} is independent of u.
        '''
        #self.percent_rankings = Rank_list(self.n_beers).rank_list
        #print (self.percent_rankings[self.n_beers])


        beers_ranks = {}
        rank =0
        #for beer, rating in self.best_beers:
        #    beers_ranks[int(beer)] = self.percent_rankings[self.n_beers][rank]
        #    rank += 1

        for beer, rating in self.best_beers:
            beers_ranks[int(beer)] = self.percent_rankings[100][rank]
            #print (rank, beer, rating, self.percent_rankings[100][rank])
            rank += 1
            if rank == 100: break

        #print (self.threshold_r)

        sum_rating_rankings = self.testData.filter(lambda x: x.product in beers_ranks).\
                              map(lambda x: x.rating*beers_ranks[x.product]).sum()


        sum_ratings = self.testData.filter(lambda x: x.product in beers_ranks).map(lambda x: x.rating).sum()

        #print (sum_rating_rankings, sum_ratings)

        #print ('after filter =>:', self.testData.filter(lambda x: x.product in beers_ranks).count())

        return sum_rating_rankings/sum_ratings


#   --------------------------------------------------------------------------


    def get_customers(self):
        customers = self.buy_records_RDD.map(lambda x: (x[0], 1)).reduceByKey(add)
        print ('number of customers = ',customers.count())

        
#   --------------------------------------------------------------------------


    def estimate_MPR(self, inputData):
        '''this function is generic to evaluate the mean-percentage-ranking
           input: 
              inputData (RDD) = (usr1, beer1, rating1), (usr2, beer2, rating2),...
        '''

        #inputData = inputData.filter(lambda x: x.rating >1)

        X_test = inputData.map(lambda x: (x.user, x.product))
        sum_ratings = inputData.map(lambda x: x.rating).sum()

        self.train_model()

        predictions_RDD = self.model.predictAll(X_test).map(lambda x: ((x[0], x[1]), x[2]))

        '''ratings_and_preds = ((user,product), (rating,predicted_probability))'''
        ratings_and_preds = inputData.map(lambda x: ((x[0], x[1]), x[2])).join(predictions_RDD)

        print ('-----------------------')
        print ('testset MPR: ', self.percentage_ranking(ratings_and_preds, sum_ratings), inputData.count())
        #print ('testset MPR: ', self.percentage_ranking(ratings_and_preds), inputData.count())
        return



#   --------------------------------------------------------------------------


    def train_model(self):
        """Train the implicit ALS model with the current dataset for A certain parameter:
        stats.stackexchange.com/questions/133565/how-to-set-preferences-for-als-implicit-feedback-in-collaborative-filtering
        """
        print(self.rank, self.seed, self.iterations, self.reg_parameter, self.alpha)
        self.model = ALS.trainImplicit(self.trainData, rank=self.rank, \
                    iterations=self.iterations, lambda_=self.reg_parameter, \
                    blocks=-1, alpha=self.alpha, nonnegative=False, seed=self.seed)
        #print('trianing is done')

#   ---------------------------------------------------------------------------

    def models_grid_search(self):
        '''the grid search on parameters for ALS model'''
        ## the ranks is the ranl of matrix factorization, min_rank_error is the metric rank
        ranks = [20]
        min_MPR = float('inf')
        best_rank = -1
        best_reg = -1.0
        best_alpha = -1.0
        best_iteration = -1
        best_model = None

        '''here we want to do grid search'''
        X_val = self.valData.map(lambda x: (x.user, x.product))
        sum_val_ratings = self.valData.map(lambda x: x.rating).sum()
        ''' sum_val_ratings = \sum_{u,i} r_{ui}, is a number, where r is in valData '''

#        '''find the maximum number of items for all users'''
#        max_n_items = X_val.map(lambda x: (x[0], [x[1]])).reduceByKey(lambda x, y: x+y).\
#                      map(lambda x: len(x[1])).max()

        #print (max_n_items)

#        '''once we know maximum number of items oruchased in the record, we can prepare
#           the rank list.
#        '''
#        rank_lists = Rank_list(max_n_items).rank_list

        for rank in ranks:
            for reg in [1.0, 2.0, 4.0]:
                for alphas in [0.1, 0.5, 1.0, 5.0, 10.0, 20.0]:

                    '''find the maximum number of items for all users
                           (usr1,[beer1]), (usr1,[beer2]), (usr2, [beer1]), (usr1,[beer3])..
                        -> (usr1,[beer1,beer2,beer3], (usr2,[beer1])..
                        -> (usr1,3), (urs2,1)...'''
                    '''once we know maximum number of items oruchased in the record, 
                           we can prepare the rank list:'''
 
                    model = ALS.trainImplicit(self.trainData, rank, iterations=self.iterations,\
                            lambda_=reg, blocks=-1, alpha=alphas, nonnegative=False, seed=self.seed)

                    '''predictions_RDD = ((user,product), predicted_probability)'''
                    predictions_RDD = model.predictAll(X_val).map(lambda x: ((x[0], x[1]), x[2]))

                    '''ratings_and_preds = ((user,product), (rating,predicted_probability))'''
                    ratings_and_preds = self.valData.map(lambda x: ((x[0], x[1]), x[2])).\
                                        join(predictions_RDD)

                    #MPR = self.percentage_ranking(ratings_and_preds)
                    MPR = self.percentage_ranking(ratings_and_preds, sum_val_ratings)

                    #break

                    print ('Rank %s, reg %s, alpha %s, AvgRank = %s' % (rank, reg, alphas, MPR))
                    if MPR < min_MPR:
                        min_MPR = MPR
                        best_model = model
                        best_rank = rank
                        best_reg = reg
                        best_alpha = alphas

        '''the grid search is done'''


        print('Best model trained by rank %s, reg %s, alpha %s' % (best_rank, best_reg, best_alpha))
        print('min ranking MPR is ', min_MPR)
        self.rank = best_rank
        self.alpha = best_alpha
        self.reg_parameter = best_reg
        self.model = best_model
        
# -----------------------------------------------------------------
    ##  MPR: mean percentage ranking
    def percentage_ranking(self, ratings_and_predictions, sum_ratings):
    #def percentage_ranking(self, ratings_and_predictions):#, sum_ratings):
        ''' This function is sued to compute average ranking.
            input: 
                  'ratings_and predictions' is a RDD of 
                  ((uer1, beer1),(rating1, prob1)), ((uer2, beer2),(rating2, prob2))....
                  ((x[0][0],x[0][1]),(x[1][0],x[1][1]))

                  'rank_lists' = [0%, xxxx.... 100%] ranking in terms of %.

            return:
            avg model_rating_rank = \sum_{u,i} r_{u,i}* rank_{u,i} / \sum_{u,i} r_{u,i}
        '''

        rank_lists = self.percent_rankings

        users_predictions = ratings_and_predictions.\
                          map(lambda x: (x[0][0], [(x[0][1], x[1][0], x[1][1])])).\
                             reduceByKey(lambda x, y: x+y)
        '''
           now 'users_predictions' is a RDD of
                (user1, [(beer11, rating11, prob11), (beer12, rating12, prob12), ...]),
                (user2, [(beer21, rating21, prob21), (beer22, rating22, prob22).....]),...
                (user, [(beers, ratings, probs)])....        
           using [a] + [b] = [a,b] in Python
        '''

        #users_predictions = users_predictions.filter(lambda x: len(x[1]) >= 5)

        #print (users_predictions.count())

        sum_ratings_rankings = users_predictions.map(lambda x: model_rank_score(x, rank_lists)).sum()
        '''  'users_prediction.map(lambda x: model_rank_score..)' gives a RDD
              \sum_i r_{ui}* rank_{ui} for each user u.
              after .sum(), it becomes to \sum_{u,i} r_{ui}*rank_{ui} 
        '''
        #users_predictions = users_predictions.map(lambda x: model_rank_score(x, rank_lists))
        #sum_ratings_rankings = users_predictions.map(lambda x: x[0]).sum()
        #sum_ratings = users_predictions.map(lambda x: x[1]).sum()



#        a = ranking_RDD.filter(lambda x: x[0] == 10926)
#        print(a.take(20), type(a.take(1)[0][0]))
#        print('')
#        b = ranking_RDD.filter(lambda x: x[0] == 163872)
#        print(b.take(20))
#        print('')
#        c = ranking_RDD.filter(lambda x: x[0] == 65574)
#        print(c.take(20))

        return sum_ratings_rankings/sum_ratings
           
#  ---------------------------------------------------------------------


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: sort <file>", file=sys.stderr)
        exit(-1)

    sc = SparkContext(appName="beers_buy_records")
    engine = ImplicitCF_engine(sc, rank=8, seed=32, iterations=20, reg_parameter=0.06)






    sc.stop()
