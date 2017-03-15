#
#
#     $$$$  using users' ratings as vectors, not beers' atrributes  $$$$
#
#               u1  u2  u3  u4 ....
#     beer-1 = (0,  0,  0,  5, 2, 0, 0, 3, 0.....)
#
#     SO, in some sense it is longer a content-based model.
#
#
import numpy as np
import pandas as pd
import operator

from time import time
#from annoy import AnnoyIndex
import random

import pymysql as mdb


local_db = mdb.connect(user="root", host="localhost", db="reco-api", charset='utf8')
source_db = mdb.connect(xxxxx)


k_neighbors=10



def cosine(a,b):
   return np.dot(a,b.T)


class CF_filter_neighborhood(object):
     '''the class for content-based filtering engine'''


     def __init__(self, k_neighbors):
         '''to initialize data we need in the engine'''

         # loading data from BeHoppy database into Python Pandas
         with source_db:
            select_query = ("select beers.id, origins.name as country, flavors.name as flavor, "
                            "styles.name as style, average_rating, number_of_ratings, "
                            "average_aroma_rating, average_appearance_rating, average_taste_rating, "
                            "average_body_rating ")
         
            from_query = ("from beers join brands on (beers.brand_id = brands.id) "
                                     "join origins on (brands.origin_id = origins.id) "
                                     "left join flavors on (beers.flavor_id = flavors.id) "
                                     "left join styles on (beers.style_id = styles.id) ")
         
            order_query = "where number_of_ratings >0 order by id"
            beers = pd.read_sql_query(select_query+from_query+order_query+";",source_db)


         beers = beers.fillna(0)
         beers = pd.get_dummies(beers)
         self.beers = beers
         ### -------  pandas preparation is done --------------


         self.best_beers = self.get_best_beers(beers)                  ## find best rated beers
         self.users_beers_ratings = self.get_users_ratings()          ## find user-beer rating records

         self.baseline_model_performance()

         ## **** this function can be off once SQL database is built, to save time. ****
         #self.users_methods_activation(k_neighbors)
         #self.beers_methods_activation(k_neighbors)
         ## *****************************************************************************

         



     def users_methods_activation(self, k_neighbors):
         '''the follows are functions not only building methods but loading data into SQL database.
            once the relevant data are load, this function can be turned off
         '''
         # -- define users' vector space
         self.vectorSpace = self.build_users_vector_space()
         # -- compute all beer-beer pair and then load in the "similarity" table'''
         self.cosineSim = self.similarity()
         # --construct user-beer recommendation table
         MAE, self.users_pred_beers_ratings = self.users_neighborhood(k_neighbors)
         print ('mean absolute error =', MAE)
         print (' ------------------------')


     def beers_methods_activation(self, k_neighbors):
         '''the follows are functions not only building methods but loading data into SQL database.
            once the relevant data are load, this function can be turned off
         '''
         # -- define beers' vector space:
         self.vectorSpace = self.build_beers_vector_space()     
         # -- compute all beer-beer pair and then load in the "similarity" table'''
         self.cosineSim = self.similarity()
         # -- construct user-beer recommendation table
         MAE, self.users_pred_beers_ratings = self.beers_neighborhood(k_neighbors)
         print ('mean absolute error =', MAE)
         print (' ------------------------')



# ----------------------------------------------------------------------------------
         

     def baseline_model_performance(self):
        '''  
           self.best_beers = [(beer1, r1, c1), (beer2, r2, c2),...]
           selfs.user_beers_ratings[user] = {beer1: r1, beer2: r2, beer3: r3....}
           ||
        '''
        avg_beer_ratings = {}
        for beer, rating, rating_counts in self.best_beers:
              avg_beer_ratings[beer] = rating


        error = 0.0
        count =0
        for user in self.users_beers_ratings:
           #print self.users_beers_ratings[user]
           for beer in self.users_beers_ratings[user]:
              if beer in avg_beer_ratings:
                 '''remind some rated beers don't show any attributes in 'beers' table.'''
                 actual_rating = self.users_beers_ratings[user][beer]
                 error += abs(actual_rating-avg_beer_ratings[beer])
                 count += 1
              else:
                 #print beer, actual_rating
                 pass

        print (error/float(count))


# ----------------------------------------------------------------------------------


     def get_best_beers(self, beers):
        ''' find top rated beers, this is the baseline model '''
        ''' NOTE!!! the best_beers list is from the 'beer' table which users rated beer's attributes, not
                    from 'rating' table. So it is possible that beers are rated (1-5) by consumers but 
                    the beers don't have attribites, like average rating of appearance, taste.....'''
        avg_beers_rating = beers[['id','average_rating','number_of_ratings']][beers.number_of_ratings > 0]
        best_beers = [tuple(x) for x in avg_beers_rating.values]
        best_beers.sort(key=lambda tup: -tup[1])

        print ('number of beers:', len(best_beers))

        return best_beers

        with local_db:
           #print 'start loading into best_beers'
           cur = local_db.cursor()
           cur.execute("DELETE FROM best_beers;")
           for beer, rating, count in best_beers:
              #print beer, type(beer), rating, type(rating), count, type(count)
              '''NOTE: all variables are in "numpy.float64" attribute.'''
              query = ("INSERT INTO best_beers "
                       "(beer, rating, count) "
                       "VALUES (%(beer)s, %(rating)s, %(count)s);")
              query_add = {'beer': int(beer), 'rating': float(rating), 'count': int(count)}
              cur.execute(query, query_add)
           cur.fetchall()
           #print 'loading best_beers is done'

        return best_beers


# ---------------------------------------------------------------------------------------


     def build_users_vector_space(self):
         '''!!! THIS PART USING USERS' RATING TO BUILD VECTOR SPACE for USERS' VECTORS!!!!
            return:
                vectorSpace[beer] = [rating from Alice, rating from Bob, rating from Carl....]
                vectorSpace[int] = [float, float, flaot,....]
         '''

         beer_index = {}
         index = -1 
         for user in self.users_beers_ratings:
            for beer in self.users_beers_ratings[user]:
               if beer not in beer_index:
                  index += 1
                  beer_index[beer] = index


#         beer_ids = self.beers[['id']]
#         index = -1
#         for beer in beer_ids.values:
#            index += 1
#            beer_index[int(beer[0])] = index
            #print beer[0], index, type(int(beer[0]))

         self.dim = len(beer_index)

         #print beer_index[5], beer_index[7]

         #return

         vectors = {}
         for user in self.users_beers_ratings:
            for beer in self.users_beers_ratings[user]:
               rating = self.users_beers_ratings[user][beer]
               if user not in vectors: vectors[user] = [0]*self.dim
               #print type(beer)
               vectors[user][beer_index[beer]] = rating


         #print 'OK?'

         #index = 0
         #for user in self.users_beers_ratings:

            #print '-----------------------'
            #print index, 'u=', user, self.users_beers_ratings[user]
            #index += 1
            #if index >= 6: break

         #   for i in range(len(vectors[user])):
         #      if vectors[user][i] !=0: print i, vectors[user][i]
      


         print ('vector space dim: ', len(vectors))

         for user in vectors:
            v1 = np.asarray(vectors[user])
            v1 = v1/np.linalg.norm(v1)
            vectors[user] = v1


         return vectors
#         pass


# ----------------------------------------------------------------------------------

   
     def build_beers_vector_space(self):
         '''this method considers loading data directly from pandas daatframe. Note that in this way
            the first column is the beer-id, and no column names will be loaded.

            !!! THIS CODE DOESN'T USE BEERS' ATRRIBUTES, INSETAD USING USERS' RATING TO BUILD A VECTOR SPACE!!!!

            return:

                vectorSpace[beer] = [rating from Alice, rating from Bob, rating from Carl,......]
            
                vectorSpace[int] =[float, float, flaot,....]
         '''

         ##  ---- define the user_id index: i.e. user_id =7, index = 1; user_id = 33, index =2...
         ##  need to find a fixed dimension representation

         vectors = {}

         avg_beers_ratings = {}

         users_index = {}
         index = -1
         for user in self.users_beers_ratings:
            #print type(user)
            index += 1
            users_index[user] = index

            #print '-----------------------'
            #print index, user, self.users_beers_ratings[user]

            #if index >= 6: break

            for beer in self.users_beers_ratings[user]:
               rating = self.users_beers_ratings[user][beer]
               if beer not in vectors: vectors[beer] = [0]*self.dim
               if beer not in avg_beers_ratings: avg_beers_ratings[beer] = []
   
               vectors[beer][index] = rating
               avg_beers_ratings[beer].append(rating)


         print ('vector space dim: ',len(vectors))



         for beer in vectors:
            #v1 = np.asarray(vectors[beer]- np.mean(avg_beers_ratings[beer]))
            v1 = np.asarray(vectors[beer])
            v1 = v1/np.linalg.norm(v1)
            vectors[beer] = v1
            #vectors[beer] = np.asarray(vectors[beer])


         #beer_id = 38

         #for i in range(len(vectors[beer_id])):
         #   if vectors[beer_id][i] !=0:
         #      print i, vectors[beer_id][i]
         #
         #return

         return vectors

      
# ----------------------------------------------------------------------------


     def similarity(self):
         '''compute the cosine distance similarity of all paired beer1-beer2 and load 
            the similarity scores in the local SQL database, "similarity" table.
            https://dev.mysql.com/doc/connector-python/en/connector-python-example-cursor-transaction.html
            return:
                 self.cosineSim[beer_i] = [(beer_1, si1), (beer_2, si2), (beer_3, si3)...]
         '''
         t0 = time()

         cosineSim = {}
         for a in self.vectorSpace:
             vector_a = self.vectorSpace[a]
             cosineSim[a] = []
             for b in self.vectorSpace:
                 vector_b = self.vectorSpace[b]
                 cosineSim[a].append((b, cosine(vector_a, vector_b)))
             x = cosineSim[a]            ## all paired items' similarity values
             x.sort(key=lambda tup: -tup[1])   ## and then sort
             cosineSim[a] = x
         
         return cosineSim

         with local_db:
             print ('start loading into similarity')
             cur = local_db.cursor()
             cur.execute("DELETE FROM similarity;")
             for beerId1 in cosineSim:
                for beerId2, score in cosineSim[beerId1]:
                   query = ("INSERT INTO similarity "
                           "(beer1, beer2, score) "
                           "VALUES (%(beer1)s, %(beer2)s, %(score)s);")
                   query_add = {'beer1': beerId1, 'beer2': beerId2, 'score': float(score)}
                   cur.execute(query, query_add)
             cur.fetchall()
             print ('loading similarity is done')

         print ('similar spends:',time()-t0,'s')

         return cosineSim
 
# -------------------------------------------------------------------------------


     def get_users_ratings(self):
         '''This function is used tp prepare users rating history''' 
         ratings = pd.read_sql_query("SELECT score,user_id, beer_id "
                                "from ratings "
                                "where user_id is not null or beer_id is not null;",source_db)
         #print ratings.head()
         '''return:

            users_beers_ratings[user] = {beer1: r1, beer2: r2, beer3: r3, beer4: r4....}

            users_beers_ratings[int] = {int: float, int:float......}

            Under this structure, one can quickly hash users in user_beers_rating 
            and hash beers in user_beers_ratings[user]
         '''
         num_ratings =0

         users_beers_ratings = {}
         for line in ratings.values:
            num_ratings += 1
            line = tuple(line)
            user_id = int(line[1])
            beer_id = int(line[2])
            score = line[0]
            if user_id not in users_beers_ratings:
               users_beers_ratings[user_id] = {beer_id:float(score)}
            else:
               users_beers_ratings[user_id][beer_id] = float(score)

         print ('number of users: ', len(users_beers_ratings))

         self.dim = len(users_beers_ratings)
         #print 'sparity = ', 1.0 - num_ratings/float(len(users_beers_ratings)*len(self.vectorSpace))


         return users_beers_ratings


# -------------------------------------------------------------------------------


     def users_neighborhood(self, k_neighbors):
         '''Using neighborhood methods to predict the ratings of unrated beers for a given
            user. k_neighbors is to control how many neighbors to include for the predictions.
            return:
              users_pred_beers_ratings[user] = [(beer1, r1), (beer2, r2), (beer3, r3)...]

            This function gurantees the predicted rating is given by summation over k-neighbors.                   
            i.e. we order the similarity, and search for the neighbor beers which have rating                      
                 until we find the most k-neighbor beers.                                                          
                 i.e. if k=4, 0 = {117:5, 2:1, 3: no rating, 10:2, 8:2, 100:1} (if desc order in similarity)      
                 so summation = sim(0,117)*5+sim(0,2)*1+ sim(0,10)*2+sim(0,8)*2         
         '''

         print ('k-NN=', k_neighbors)

         t0 =time()

         error = 0.0
         repeat_pred = 0

         ## beers appeared in rating table (but can have no attribute evaluations, so not appear in beer table)
         beers_set = set()
         for user in self.users_beers_ratings:
            for beer in self.users_beers_ratings[user]:
               if beer not in beers_set: beers_set.add(beer)


         users_pred_beers_ratings = {}

         #irow =0
         for user1 in self.cosineSim:

            #irow += 1
            #if irow >= 4: break

            #print '----------------------------------'
            #print 'user1= ', user1

            #jrow =0 
            for beer in beers_set:
               #jrow += 1

               #if jrow >= 4: break

               #print '***************'
               #print 'beer=', beer

               beer_pred_rating =0.0
               norm_sim = 0.0
               for user2, similarity in self.cosineSim[user1][1:k_neighbors+1]:
                  #print '********'
                  #print user2, similarity, beer in self.users_beers_ratings[user2]

                  if beer in self.users_beers_ratings[user2]:
                     rating = self.users_beers_ratings[user2][beer]
                     beer_pred_rating += similarity*rating
                     norm_sim += abs(similarity)

               '''as above, if norm_sim=beer_pred_rating=0: no predicted rating available'''
               if norm_sim == 0 or beer_pred_rating < 0:
                  beer_pred_rating = 0
               else:
                  beer_pred_rating = beer_pred_rating/norm_sim


               if beer_pred_rating >0:
                  #print 'pred', user1, beer, beer_pred_rating, norm_sim
                  if beer not in self.users_beers_ratings[user1]:
                     if user1 not in users_pred_beers_ratings:
                        users_pred_beers_ratings[user1] = [(beer, beer_pred_rating)]
                     else:
                        users_pred_beers_ratings[user1].append((beer, beer_pred_rating))
                  else:
                     error += abs(beer_pred_rating-self.users_beers_ratings[user1][beer])
                     repeat_pred += 1
                     #print 'repeat', beer_pred_rating, self.users_beers_ratings[user1][beer]


         '''users_pred_beers_ratings[user] = [(beer1, r1), (beer2, r2), (beer3, r3)...]
            now, we need to put it in order according to ratings, r1>= r2>=....:
         '''
         for user in users_pred_beers_ratings:
            #print user, users_pred_beers_ratings[user]
            #print 
            x = users_pred_beers_ratings[user]
            x.sort(key=lambda tup: -tup[1])
            users_pred_beers_ratings[user] = x


         #print users_pred_beers_ratings


         return error/float(repeat_pred), users_pred_beers_ratings


         with local_db:
            print ('start loading into usersReco')
            cur = local_db.cursor()
            cur.execute("DELETE FROM usersReco;")
            for user in users_pred_beers_ratings:
               for beer, rating in users_pred_beers_ratings[user][:50]:
                  #print user, type(user), int(beer), type(int(beer)),rating
                  '''NOTE: doing int(beer), otherwise beer is in "numpy.int64" attribute.'''
                  query = ("INSERT INTO usersReco "
                           "(user_id, beer_id, pred_rating) "
                           "VALUES (%(user_id)s, %(beer_id)s, %(pred_rating)s);")
                  query_add = {'user_id': user, 'beer_id': int(beer), 'pred_rating': float(rating)}
                  cur.execute(query, query_add)
            cur.fetchall()
            print ('loading usersReco is done')

         print ('neighborhood spends:',time()-t0,'s')

 
         return error/float(repeat_pred), users_pred_beers_ratings


# -------------------------------------------------------------------------------


     def beers_neighborhood(self, k_neighbors):
         '''Using neighborhood methods to predict the ratings of unrated beers for a given
            user. k_neighbors is to control how many neighbors to include for the predictions.
            return:
              users_pred_beers_ratings[user] = [(beer1, r1), (beer2, r2), (beer3, r3)...]

            This function considers the predicted rating is given by summation over k-neighbors with ratings.
            i.e. we always targe k-neihgbor points, but if the neighbor beers don't have rating,
                 we don't put in the summation.
                 i.e. if k=4, 0 = {117:5, 2:1, 3: no rating, 10:2, 8:2, 100:1} (if desc order in similarity)
                 so summation = sim(0,117)*5 + sim(0,2)*1 + sim(0,10)*2
         '''

         print ('k-NN=', k_neighbors)

         t0 =time()

         error = 0.0
         repeat_pred = 0


         users_pred_beers_ratings = {}


#         irow =0
         for beer1 in self.cosineSim:

#            irow += 1
#            if irow >= 6: break

            #print '-------------------'
            #print 'beer1', beer1

            weight = {}
            for beer2, similarity in self.cosineSim[beer1][1:k_neighbors+1]:
               weight[beer2] = similarity#1.0 - similarity/2.0


#            jrow= 0
            for user in self.users_beers_ratings:

#               jrow += 1
#               if jrow >=4: break
              
               #print 'user', user, self.users_beers_ratings[user]

               beer1_pred_rating =0.0
               norm_sim = 0.0

               for beer2 in self.users_beers_ratings[user]:
                  if beer2 in weight:
                     '''Note we will encounter TWO situations where no beer_2 in similarity_weight: 
                        (1) beer_2 has rating by users, but has not been characterized attributes yet, 
                            i.e. in "beers" table it doesn't exist. So in the similarity table as well as
                            similarity_weight, no beer1-beer2 record exists.
                        (2) beer2 has been characterized attributes in "beers" table, so has beer1-beer2 
                            similairty. However, the similarity is not significant such that beer-2 is not 
                            in the set of k-neighbors.
                        In both cases, we will have norm_sim = beer1_pred_rating = 0
                      '''

                     rating = self.users_beers_ratings[user][beer2]
                     #print user, beer1, beer_2, rating, weight[beer2]
                     beer1_pred_rating += weight[beer2]*rating
                     norm_sim += abs(weight[beer2])
                     '''the above procedures are doing sum_j (s_{ij}*r_{uj}) and sum_j |s_{ij}|'''

               #print 'pred', beer1_pred_rating, norm_sim
               '''as above, if norm_sim=beer1_pred_rating=0: no predicted rating available'''
               if norm_sim == 0 or beer1_pred_rating < 0:
                  beer1_pred_rating = 0
               else:
                  beer1_pred_rating = beer1_pred_rating/norm_sim

               #print 'pred', beer1, beer1_pred_rating, norm_sim

               if beer1_pred_rating >0:
                  if beer1 not in self.users_beers_ratings[user]:
                     if user not in users_pred_beers_ratings:
                        users_pred_beers_ratings[user] = [(beer1, beer1_pred_rating)]
                     else:
                        users_pred_beers_ratings[user].append((beer1, beer1_pred_rating))
                  else:
                     error += abs(beer1_pred_rating-self.users_beers_ratings[user][beer1])
                     repeat_pred += 1
                     #print 'repeat', beer1_pred_rating, self.users_beers_ratings[user][beer1]


         '''users_pred_beers_ratings[user] = [(beer1, r1), (beer2, r2), (beer3, r3)...]
            now, we need to put it in order according to ratings, r1>= r2>=....:
         '''
         for user in users_pred_beers_ratings:
            x = users_pred_beers_ratings[user]
            x.sort(key=lambda tup: -tup[1])
            users_pred_beers_ratings[user] = x


         #print users_pred_beers_ratings


         return error/float(repeat_pred), users_pred_beers_ratings


         with local_db:
            print ('start loading into usersReco')
            cur = local_db.cursor()
            cur.execute("DELETE FROM usersReco;")
            for user in users_pred_beers_ratings:
               for beer, rating in users_pred_beers_ratings[user][:50]:
                  #print user, type(user), int(beer), type(int(beer)),rating
                  '''NOTE: doing int(beer), otherwise beer is in "numpy.int64" attribute.'''
                  query = ("INSERT INTO usersReco "
                           "(user_id, beer_id, pred_rating) "
                           "VALUES (%(user_id)s, %(beer_id)s, %(pred_rating)s);")
                  query_add = {'user_id': user, 'beer_id': int(beer), 'pred_rating': float(rating)}
                  cur.execute(query, query_add)
            cur.fetchall()
            print ('loading usersReco is done')

         print ('neighborhood spends:',time()-t0,'s')

         return error/float(repeat_pred), users_pred_beers_ratings


# -------------------------------------------------------------------------------


     def annoySearch(self):
         '''implement annoy library for approx NN search
            https://github.com/spotify/annoy
            http://www.slideshare.net/erikbern/approximate-nearest-neighbor-methods-and-vector-models-nyc-ml-meetup
         '''
         f = self.dim
         print (f)
         t = AnnoyIndex(f)  # Length of item vector that will be indexed
         i =0
         self.annoyId_beerId = {}
         self.beerId_annoyId = {}
         for beer_id in self.vectorSpace:
            self.annoyId_beerId[i] = beer_id
            self.beerId_annoyId[beer_id] = i
            v = self.vectorSpace[beer_id]
            t.add_item(i, v)
            i += 1

         t.build(10) # 10 trees
         t.save('test.ann')


# ----------------------------------------------------------------------------





#a = [500]#[100,120,150,200,250,300,350,400,450,500]

CB_engine = CF_filter_neighborhood(k_neighbors)

#for k_neighbors in a:
#   CF_filter_neighborhood(k_neighbors)



#exit()




### ---- THE FOLLOWINGS ARE DESIGNED FOR UI QUERY -----


user = 7

print ('user_id=', user)


cursor = local_db.cursor()
cursor.execute("select * from usersReco where user_id="+str(user)+" limit 10;")

if not cursor.rowcount:
   print (' ---- new users!! recommend best beers: ----')
   best_beers=pd.read_sql_query("select beer, rating, count "
                            "from best_beers where count > 50 limit 10;", local_db)
   print (best_beers.head(10))
else:
   pred_beers=pd.read_sql_query("select user_id, beer_id, pred_rating "
                               "from usersReco "
                              "where user_id="+str(user)+" limit 10;", local_db)
   print (pred_beers.head(10))





#for beer, rating in CB_engine.users_pred_beers_ratings[user][:10]:
#   print (user, beer, rating)


#print CB_engine.cosineSim[210]

# ------------------------------------------------------------------


beer_item = 55




## beers which are similar to beer-id:
## ----- (1) using ANNOY to search -------

#t0 = time()
#CB_engine.annoySearch()
#u = AnnoyIndex(CB_engine.dim)
#u.load('test.ann')


#item = CB_engine.beerId_annoyId[beer_item]
#a = u.get_nns_by_item(item, 20)
##print a
#for i in a[:10]:
#   print CB_engine.annoyId_beerId[item], CB_engine.annoyId_beerId[i], u.get_distance(item, i)
#print 'ANNOY spends:', time()-t0, 's'


## ------ (2) using SQL query to list the rank --------

print ('')
print ('beer:', beer_item)


t0 = time()

item_reco=pd.read_sql_query("select beer1, beer2, score "
                            "from similarity where beer1 = "+str(beer_item)+" limit 10;", local_db)

print (item_reco.head(10))


#for beer, score in CB_engine.cosineSim[beer_item][:20]:
#   print (beer_item, beer, round(score,3))



#for line in item_reco.values[:10]:
#   print line[0], line[1], line[2]

#print 'SQL reading similarity spends:', time()-t0, 's'

#print 


#for beer2, score in CB_engine.cosineSim[beer_item][:10]:
#   print beer_item, beer2, score
#print


## ------- top-5 rated beers  --------------------

#best_beers=pd.read_sql_query("select beer, rating, count "
#                            "from best_beers limit 10;", local_db)
#print best_beers.head(10)

#for beer, rating, count in CB_engine.best_beers[:20]:
#   print int(beer), rating, int(count)


