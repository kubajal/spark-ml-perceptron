import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.sql.{SQLContext, SparkSession}

trait Configuration {

  val conf = new SparkConf()
  conf.setMaster("local[*]")
  conf.setAppName("WEDT")
  val sparkSession: SparkSession = SparkSession
    .builder
    .config(conf)
    .getOrCreate()
  val sparkContext: SparkContext = sparkSession.sparkContext
  sparkContext.setLogLevel("OFF")
  val defaultPath = "resources/20-newsgroups/*"
  val sqlContext: SQLContext = sparkSession.sqlContext
  val inputNeurons = 54
  val outputNeurons = 41
}