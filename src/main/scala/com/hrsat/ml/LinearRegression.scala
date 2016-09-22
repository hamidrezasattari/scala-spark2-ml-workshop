package com.hrsat.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

/**
  *  Created by hamid on 20/09/16.
  *  houses.zip source:  http://lib.stat.cmu.edu/datasets/
    These spatial data contain 20,640 observations on housing prices with 9 economic covariates.
    The file cadata.txt contains all the the variables. Specifically, it contains median house value,
    median income, housing median age,   total rooms, total bedrooms, population, households, latitude,
    and longitude in that order.
  */
object LinearRegression {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("linear-regression")
      .getOrCreate()

    val data = spark.read.format("libsvm").load("data/cadata")


    val lr = new LinearRegression()


    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    // Split the data into training and test sets (10% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1), seed = 12345)

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 90% of the data will be used for training and the remaining 10% for validation.
      .setTrainRatio(0.9)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(trainingData)

    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    model.transform(testData)
      .select("features", "label", "prediction")
      .show()


  }


  }
