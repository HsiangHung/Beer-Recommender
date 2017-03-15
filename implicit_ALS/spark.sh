#!/bin/bash


spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/emporio.csv > test_26

spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/emporio.csv > test_27

spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/emporio.csv > test_28

spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/emporio.csv > test_29

spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/emporio.csv > test_30


#spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/recent.csv > test


#spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/more_recent.csv > test

#spark-submit --master spark://ip-172-31-11-19:7077 als_implicit.py hdfs://54.86.54.194:9000/test/full.csv > test
