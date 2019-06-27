spark-submit --class org.apache.spark.examples.JavaWordCount --master yarn --deploy-mode client --executor-memory 1g spark-examples_2.10-assembly-0.9.1.jar hdfs://localhost:9000/test/input/words

#spark-submit --class org.apache.spark.examples.JavaWordCount --master yarn --deploy-mode cluster --executor-memory 1g spark-examples_2.10-assembly-0.9.1.jar hdfs://localhost:9000/test/input/words

