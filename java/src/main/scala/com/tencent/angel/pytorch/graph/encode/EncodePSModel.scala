package com.tencent.angel.pytorch.graph.encode

import com.tencent.angel.ml.matrix.{MatrixContext, RowType}
import com.tencent.angel.psagent.PSAgentContext
import com.tencent.angel.spark.models.PSVector
import com.tencent.angel.spark.models.impl.PSVectorImpl
import com.tencent.angel.ml.math2.vector.{LongIntVector, Vector}

private[encode]
class EncodePSModel(index: PSVector) extends Serializable {

  val dim: Long = index.dimension

  def initIndex(update: Vector): Unit =
    index.update(update)

  def readIndex(nodes: Array[Long]): LongIntVector =
    index.pull(nodes).asInstanceOf[LongIntVector]

}

private[encode]
object EncodePSModel {
  def fromMinMax(minId: Long, maxId: Long): EncodePSModel = {
    val index = new MatrixContext("index", 1, minId, maxId)
    index.setRowType(RowType.T_INT_SPARSE_LONGKEY)
    index.setValidIndexNum(-1)

    PSAgentContext.get().getMasterClient.createMatrix(index, 1000L)
    val indexId = PSAgentContext.get().getMasterClient.getMatrix("index").getId
    new EncodePSModel(
      new PSVectorImpl(indexId, 0, maxId, index.getRowType)
    )
  }
}
