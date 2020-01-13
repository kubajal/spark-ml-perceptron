import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.sql.DataFrame

class MulticlassClassifier(lays: List[Int], train: DataFrame) extends MultilayerPerceptronClassifier with Configuration {

    this.setLayers(lays.toArray)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
  val model: Model[_] = this.fit(train)

  def transform(df: DataFrame): DataFrame = this.model.transform(df)
}