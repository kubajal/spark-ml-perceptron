import java.io.{File, PrintWriter}
import java.time.Instant

import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object WMH extends App with Configuration with scalax.chart.module.Charting {

  import sqlContext.implicits._

  var labelIndexerModel: StringIndexerModel  = _

  def prepareRdd(rdd: RDD[String]) = {
    val df1 = rdd
      .flatMap(e => e.split("\n"))

    df1.map(e => e.split("  | ").tail)
      .map(e => e.map(f => f.toDouble))
      .map(e => (Vectors.dense(e.dropRight(1)), e.last))
      .toDF("features", "label_0")
      .persist
  }

  def combine(in: List[Char]): Seq[String] =
    for {
      len <- 1 to in.length
      combinations <- in combinations len
    } yield combinations.mkString

  def testCase(layer: List[Int], stepSize: Double, trainDf: DataFrame, testDf: DataFrame, metric: String): Unit = {
    val classifier = new MultilayerPerceptronClassifier("WMH")
    val paramMap = new ParamGridBuilder()
      .addGrid(classifier.layers, Array(layer.toArray))
      .addGrid(classifier.stepSize, Array(stepSize))
      .build()
    val crossValidator = new CrossValidator()
      .setEstimator(classifier)
      .setEvaluator(new MulticlassClassificationEvaluator())
      .setEstimatorParamMaps(paramMap)
      .setNumFolds(5)

    val start = Instant.now()
    val trainedModel = crossValidator.fit(testDf)
    val stop = Instant.now()

    val result = trainedModel.transform(testDf)

    val metrics = new MulticlassMetrics(result.rdd
      .map(row => (row.getAs[Double]("prediction"), row.getAs[Double]("label"))))

    val time = stop.toEpochMilli - start.toEpochMilli

    println(s"label,precision,recall")
    (0.0 to 40.0 by 1.0).foreach(e => {
      val label = labelIndexerModel.labelsArray(0)(e.toInt)
      val precision = metrics.precision(e)
      val recall = metrics.recall(e)
      println(s"$label,$precision,$recall")
    })

    println(metrics.labels)
    val m = metrics.confusionMatrix.rowIter.map(e => e.toArray.map(f => f.toInt).toVector)
    m.foreach(e => println(e))

    //println(s"${layer(1)},${trainedModel.avgMetrics.head},$time,$stepSize")
  }

  override def main(args: Array[String]): Unit = {

      val train = sparkContext.textFile("resources/rs_test5_exhaustive_no_headers.txt")
      val test = sparkContext.textFile("resources/rs_test5_exhaustive_no_headers.txt")

      val labelIndexer = new StringIndexer()
        .setInputCol("label_0")
        .setOutputCol("label")

      val trainDf1 = this.prepareRdd(train)
      val testDf1 = this.prepareRdd(test)
      labelIndexerModel = labelIndexer.fit(trainDf1)
      val trainDf = labelIndexerModel.transform(trainDf1)
      val testDf = labelIndexerModel.transform(testDf1)
      val metric = "precision"

      println(s"wczytano ${trainDf.count} wierszy dla trainDf")
      println(s"wczytano ${testDf.count} wierszy dla testDf")

      println(s"layer(1),f1,precision,recall,czas,stepSize")
      val layers = for {
        x <- 70 to 80 by 10
        y <- 70 to 90 by 10
      } yield List(inputNeurons, x, y, outputNeurons)

      for {
        stepSize <- List(0.7)
        layer <- layers
      } yield {
        testCase(layer, stepSize, trainDf, testDf, metric)
      }
    }

    sparkSession.close()

}
