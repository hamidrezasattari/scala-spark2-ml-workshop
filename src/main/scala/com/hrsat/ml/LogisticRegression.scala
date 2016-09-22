package com.hrsat.ml

import org.apache.log4j.{Level, Logger}

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.functions._

/**
  * Created by hamid on 21/09/16.
  *  dataset: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/ref.html#AVU06a
  */
object LogisticRegression {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("decision-tree-regression")
      .getOrCreate()
    import spark.implicits._

    val data = spark.read.format("libsvm").load("data/cod-rna").withColumn("label", when($"label" ===  -1 , 0).otherwise(1)).cache()

     val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1))

    val lr = new LogisticRegression()



     println("LogisticRegression default parameters:\n" + lr.explainParams() + "\n")

    // We   set parameters using setters
    lr.setMaxIter(10)
      .setRegParam(0.01)
    val model1 = lr.fit(trainingData)

    println("Model 1   parameters: " + model1.parent.extractParamMap)




    //change parameters using ParamMap
    val paramMap = ParamMap()
      .put(lr.maxIter,30)
      .put(lr.regParam -> 0.1, lr.threshold -> 0.55)




    val model2 = lr.fit(data, paramMap)
    println("Model 2  parameters: " + model2.parent.extractParamMap)




    model2.transform(testData) .show()


  }
}
