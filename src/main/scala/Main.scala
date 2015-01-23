import breeze.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
 * Created by karthik on 1/22/15.
 * Perceptron algorithm on spark
 */

import breeze.linalg.DenseVector
case class Point(y: Double, x: DenseVector[Double])

object Main {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster(args(0))
    val sc = new SparkContext(conf)
    val pointFile = args(1)
    val MAXITERATIONS = args(2).toInt
    //val hyperPlane = new DenseVector[Double](args(3).split(" ").map(_.toDouble).toArray)

    val pointsFile = sc.textFile(pointFile)

    // this is cached, so avoid reading from disk again. :)
    val pointsRDD = pointsFile.map{ line =>
      val data = line.split(";")
      val label = data(0).trim.toDouble
      val point = data(1).trim.split(" ").map(_.toDouble).toArray
      Point(label , new DenseVector[Double](point))
    }.cache()

    pla(pointsRDD, MAXITERATIONS)
    sc.stop()
  }

  def pla(pointsRDD: RDD[Point], MAXITERATIONS: Int) = {
    var i = 0
    var w: DenseVector[Double] = new DenseVector[Double](Array.fill(11)(0))
    var isConverged = false

    while(i < MAXITERATIONS && isConverged == false) {

      //transformation
      val incorrect_points = pointsRDD.filter { point =>
        val h_x:Double = dot(w, point.x)
        val guessedLabel = if(h_x >= 0) 1.0 else -1.0
        guessedLabel != point.y
      }
      isConverged = incorrect_points.count <= 0
      if(!isConverged) {
        //action
        val point = incorrect_points.first
        w = add(w, mul(point.x, point.y))
      }
      i += 1
    }
    println(s"The learnt hyperplane is ${w} in ${i} iterations")
  }

  def dot(x: DenseVector[Double], y: DenseVector[Double]) = {
    (0 to x.length -1).map(i => x(i) * y(i))
    .sum
  }

  def add(x: DenseVector[Double], y: DenseVector[Double]) = {
    val a = (0 to x.length -1).map(i => x(i) + y(i)).toArray
    new DenseVector[Double](a)
  }

  def mul(x: DenseVector[Double], k: Double) = {
    x.map(k*_)
  }
}