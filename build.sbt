name := "spark2-ml-workshop"

version := "1.0"
val sparkVersion= "2.0.0"
scalaVersion := "2.11.8"
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
   "org.apache.spark" %% "spark-mllib"  % sparkVersion,
   "log4j" % "log4j" % "1.2.14"
)





