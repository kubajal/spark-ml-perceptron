object Experiment extends Configuration {

  type Layer = Array[Int]

  object LayersGenerator {

    def generate(n: Int, x: Int, a: List[List[Int]]): List[List[Int]] = {
      val aa = a.map(e => List.range(1, x).flatMap(f => f :: e))
      if(n > 0) {
        generate(n - 1, x, aa)
      }
      else
        aa
    }

  }

}
