import java.io.{File, PrintWriter}

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, MultilayerPerceptronClassifier, NaiveBayes}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import scala.util.{Failure, Success, Try}

object WMH extends App with Configuration with scalax.chart.module.Charting {

  def prepareRdd(path: String): RDD[(org.apache.spark.ml.linalg.Vector, Double)] = {

    val plainTextTry = Try(sparkContext.textFile(path))
    plainTextTry match {
      case Success(plainText) =>

        println("liczba wczytanych linii: " + plainText.count())

        plainText
          .map(e => e.split("  | ").tail)
          .map(e => e.map(f => f.toDouble))
          .map(e => (Vectors.dense(e.dropRight(1)), labelsMap(e.last + 110)))
      case Failure(e) =>
        println(s"Could not load files from the path: $path")
        sparkContext.stop()
        throw e
    }
  }

  def combine(in: List[Char]): Seq[String] =
    for {
      len <- 1 to in.length
      combinations <- in combinations len
    } yield combinations.mkString

  override def main(args: Array[String]): Unit = {

    import sqlContext.implicits._
//
//    val e = Experiment.LayersGenerator.generate(3, 3, List.range(1, 3).map(e => List(e)))
//
//    println(e)

    val path = if(args.length == 0) "resources/*" else args.head

    val df = this.prepareRdd(path)
      .toDF("features", "label")

    println("length: " + df.collect.head.getAs[org.apache.spark.ml.linalg.Vector]("features").size)
    val labels = df.map(e => e.getAs[Double]("label")).collect.toList
    println("total number of labels: " + labels.distinct.size)
    println("labels: " + labels.distinct)

    df.printSchema()

    df.show(false)
    println("liczba linii: " + df.count)

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
    val precisionEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("weightedPrecision")

    val experiments = for {
      x <- 1 to 10
    } yield List(inputNeurons, x, outputNeurons)

    val result = experiments.flatMap(e => {
      (1 to 2).map(f => {
        val Array(train, validate) = df
          .randomSplit(Array(0.8, 0.2))
        val classifier = new MulticlassClassifier(e, train)
        val predictions = classifier.transform(validate)
        val accuracy = accuracyEvaluator.evaluate(predictions)
        val precision = precisionEvaluator.evaluate(predictions)
        ((e(1), accuracy), (e(1), precision))
        println(s"Layers    = $e")
        println(s"Accuracy  = $accuracy")
        println(s"Precision = $precision")
        ((e(1), accuracy), (e(1), precision))
      })
    })


    val (accuracy, precision): (Seq[(Int, Double)], Seq[(Int, Double)]) = result.unzip

    val writer1 = new PrintWriter(new File("tmp/accuracy.txt"))
    val writer2 = new PrintWriter(new File("tmp/precision.txt"))

    accuracy.foreach(e => writer1.println(s"${e._1};${e._2}"))
    writer1.close()
    precision.foreach(e => writer2.println(s"${e._1};${e._2}"))
    writer2.close()

    val chart1 = XYLineChart(accuracy)
    chart1.plot.setRenderer(new org.jfree.chart.renderer.xy.XYLineAndShapeRenderer(false, true))
    chart1.saveAsPNG("tmp/accuracy.png")
    val chart2 = XYLineChart(precision)
    chart2.plot.setRenderer(new org.jfree.chart.renderer.xy.XYLineAndShapeRenderer(false, true))
    chart2.saveAsPNG("tmp/precision.png")

    sparkSession.close()
  }
}
