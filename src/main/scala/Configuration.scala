import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{SQLContext, SparkSession}

trait Configuration {

  val labelsMap = Map(
    99.0 -> 0.0,
    110.0  -> 1.0,
    121.0 ->  2.0,
    89.0 -> 3.0,
    131.0 -> 4.0,
    79.0 -> 5.0,
    141.0 -> 6.0,
    69.0 -> 7.0,
    151.0 -> 8.0,
    59.0 -> 9.0,
    161.0 -> 10.0,
    49.0 -> 11.0,
    171.0 -> 12.0,
    39.0 -> 13.0,
    181.0 -> 14.0,
    29.0 -> 15.0,
    191.0 -> 16.0,
    19.0 -> 17.0,
    201.0 -> 18.0,
    9.0 -> 19.0,
    211.0 -> 20.0)

  val conf = new SparkConf()
  conf.setMaster("local")
  conf.setAppName("WEDT")
  val sparkSession: SparkSession = SparkSession
    .builder
    .config(conf)
    .getOrCreate()
  val sparkContext: SparkContext = sparkSession.sparkContext
  sparkContext.setLogLevel("ERROR")
  val defaultPath = "resources/20-newsgroups/*"
  val sqlContext: SQLContext = sparkSession.sqlContext
  val hiddenLayersMax = 5
  val inputNeurons = 54
  val outputNeurons = 21
}