import java.io.{File, PrintWriter}
import java.time.Instant

import org.apache.spark.ml.{Model, Pipeline}
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object WMH extends App with Configuration with scalax.chart.module.Charting {

  import sqlContext.implicits._

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

  def testCase(layer: List[Int], stepSize: Double, trainDf: DataFrame, testDf: DataFrame): Unit = {
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
    val trainedModel = crossValidator.fit(trainDf.union(testDf))
    val stop = Instant.now()

    val time = stop.toEpochMilli - start.toEpochMilli

    println(s"${layer(1)};${layer(2)};${trainedModel.avgMetrics(0)};$time;$stepSize")
  }

  override def main(args: Array[String]): Unit = {

    val train = sparkContext.textFile("resources/rs_test5_exhaustive_no_headers.txt")
    val test = sparkContext.textFile("resources/rs_test5_exhaustive_no_headers.txt")

    val labelIndexer = new StringIndexer()
      .setInputCol("label_0")
      .setOutputCol("label")

    val trainDf1 = this.prepareRdd(train)
    val testDf1 = this.prepareRdd(test)
    val labelIndexerModel = labelIndexer.fit(trainDf1)
    val trainDf = labelIndexerModel.transform(trainDf1)
    val testDf = labelIndexerModel.transform(testDf1)

    println(s"wczytano ${trainDf.count} wierszy dla trainDf")
    println(s"wczytano ${testDf.count} wierszy dla testDf")

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    println(s"layer(1);layer(2);f1;czas;stepSize")
    val layers = for {
      x <- 10 to 100 by 10
      y <- 10 to 100 by 10
    } yield List(inputNeurons, x, y, outputNeurons)

    testCase(List(inputNeurons, 80, 80, outputNeurons), 0.7, trainDf, testDf)

    // two layers tests:
//    for {
//      stepSize <- 0.1 to 1.00 by 0.3
//      layer <- layers
//    } yield {
//      testCase(layer, stepSize, trainDf, testDf)
//    }

  //        classifier
  //          .setLayers(e.toArray)
  //          .setMaxIter(250)
  //          .fit(trainDf)
  //
  //      val result = cv.transform(testDf)
  //
  //      val accuracy = accuracyEvaluator.evaluate(result)
  //      val precission = precisionEvaluator.evaluate(result)
  //
  //      println(s"Accuracy for test ${e(1)}/${e(2)}   = $accuracy")
  //      println(s"Precission for test ${e(1)}/${e(2)} = $precission")

  //      val cv = new CrossValidator()
  //        .setEstimator(classifier)
  //        .setEvaluator(new MulticlassClassificationEvaluator())
  //        .setEstimatorParamMaps(Array(paramMap))
  //        .setNumFolds(5)
  //        .fit(df)
  //      cv.avgMetrics
  //        .foreach(f => println(s"metrics for test ${e(1)}: $f"))
    }

    sparkSession.close()

//    val (accuracy, precision): (Seq[(Int, Double)], Seq[(Int, Double)]) = result.unzip
//
//    val writer1 = new PrintWriter(new File("tmp/accuracy.txt"))
//    val writer2 = new PrintWriter(new File("tmp/precision.txt"))
//
//    accuracy.foreach(e => writer1.println(s"${e._1};${e._2}"))
//    writer1.close()
//    precision.foreach(e => writer2.println(s"${e._1};${e._2}"))
//    writer2.close()
//
//    val chart1 = XYLineChart(accuracy)
//    chart1.plot.setRenderer(new org.jfree.chart.renderer.xy.XYLineAndShapeRenderer(false, true))
//    chart1.saveAsPNG("tmp/accuracy.png")
//    val chart2 = XYLineChart(precision)
//    chart2.plot.setRenderer(new org.jfree.chart.renderer.xy.XYLineAndShapeRenderer(false, true))
//    chart2.saveAsPNG("tmp/precision.png")
//
}
