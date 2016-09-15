package com.hrsat.ml

import org.apache.log4j._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DataTypes, StructField, StructType}

//recommendation.ALS spark 2.0 for prediction task
// data comes from ml-100k.zip http://grouplens.org/datasets/movielens/
// u.data is modified by adding  user id 0 and then   how model prediction of this user
// for new movie by   spark ml recommendation ALS

object SparkMlRecommendationALS {


  def main(args: Array[String]) {

    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)

    val spark = SparkSession
      .builder
      .appName("spark_ml_recommendation_ALS_for_movies")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._

    val userId = StructField("userId", DataTypes.IntegerType)
    val movieId = StructField("movieId", DataTypes.IntegerType)
    val rating = StructField("rating", DataTypes.FloatType)


    val fields = Array(userId, movieId, rating)
    val schema = StructType(fields)
    val colNames = Seq("userId", "movieId", "rating")

    val ratings = spark.read.
      option("header", false).option("delimiter", "\\t")
      .schema(schema)
      .csv("data/u_added_0_userId.data")
      .map(x => Rating(x.getAs[Int](0), x.getAs[Int](1), x.getAs[Float](2)))
      .toDF(colNames: _*).cache()
    ratings.printSchema()
    // 3 added data for user 0
    ratings.take(3).foreach(println)

    val als = new ALS()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(ratings)

    // create 3 new record and assume  user not rated (with -1 values)   them (two user new movies : 172, 133), and
    // for movie : 50 , prediction should be almost same as  the user  already rated (0,50,2).
    // result predictions shows how model predict base on Collaborative filtering


    val userRatings = Seq(
      Rating(0, 50, -1f),
      Rating(0, 172, -1f),
      Rating(0, 133, -1f)
    ).toDF(colNames: _*)

    val predictions = model.transform(userRatings)
    predictions.show()



    spark.stop()


  }
}