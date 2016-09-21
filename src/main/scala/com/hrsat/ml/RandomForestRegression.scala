package com.hrsat.ml

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}

/**
  *  Created by hamid on 20/09/16.
  *  houses.zip source:  http://lib.stat.cmu.edu/datasets/
    These spatial data contain 20,640 observations on housing prices with 9 economic covariates.
    The file cadata.txt contains all the the variables. Specifically, it contains median house value,
    median income, housing median age,   total rooms, total bedrooms, population, households, latitude,
    and longitude in that order.
  */
object RandomForestRegression {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName("random-forest-regression")
      .getOrCreate()

    val data = spark.read.format("libsvm").load("data/cadata")

    // Automatically identify categorical features, and index them.
    // Here, we treat features with > 4 distinct values as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(6)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(training, testing) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val randomForest = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, randomForest))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(training)

    // Make predictions.
    val predictions = model.transform(testing)

    // Select example rows to display.
    predictions.select("prediction", "label", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    val randomForestRegressionModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
    println("Learned regression tree model:\n" + randomForestRegressionModel.toDebugString)


  }


  }
