#!/bin/bash

python BeHoppy_query.py

hdfs dfs -rm /test/Behoppy_ratings

hdfs dfs -copyFromLocal beHoppy_ratings.csv /test/Behoppy_ratings

spark-submit --master spark://xxxx als_explicit_rating.py hdfs://xxxx/test/Behoppy_ratings > test

<<<<<<< HEAD
=======

>>>>>>> origin/master
