package com.hrsat.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

/**
  * Created by hamid on 19/09/16.
  * Neural Net: multilayer perceptron  classification for pen digit data set: datasource:http://archive.ics.uci.edu/ml/
  */
object MlpClassifier {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("multilayer_perceptron_classification")
      .getOrCreate()

    val training  = spark.read
    .option("header", false).option("delimiter", " ")
      .format("libsvm").load("data/pendigits")
    val testing  = spark.read
      .option("header", false).option("delimiter", " ")
      .format("libsvm").load("data/pendigits")
    training.take(5).foreach(println)
 //[8.0,(16,[0,1,2,3,4,5,6,9,10,11,12,13,14,15],[47.0,100.0,27.0,81.0,57.0,37.0,26.0,23.0,56.0,53.0,100.0,90.0,40.0,98.0])]
 // 10 class/lable (0-9) and 16 input parameter makes Mlp with 16 input and 10 output

    val   layers =   Array[Int](16, 20, 20, 10)
    val trainer =  new MultilayerPerceptronClassifier()
      .setBlockSize(128).setLayers(layers).setMaxIter(100)
     val model = trainer.fit(training)
    // compute accuracy on the test set
    val result = model.transform(testing)
    result.printSchema()
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    println("Accuracy: " + evaluator.evaluate(predictionAndLabels))  }
}
