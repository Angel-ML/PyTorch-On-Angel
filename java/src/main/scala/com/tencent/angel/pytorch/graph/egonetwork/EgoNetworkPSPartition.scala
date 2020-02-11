package com.tencent.angel.pytorch.graph.egonetwork

import breeze.numerics.abs
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.longs.{LongArrayList, LongOpenHashSet}

import scala.collection.mutable.ArrayBuffer

private[egonetwork]
class EgoNetworkPSPartition(index: Int,
                            keys: Array[Long],
                            indptr: Array[Int],
                            neighbors: Array[Long],
                            indexGap: Long) extends Serializable {

  def init(model: EgoNetworkPSModel, numBatch: Int): Int = {
    model.initNeighbors(keys, indptr, neighbors, numBatch)
    0
  }

  def sample(model: EgoNetworkPSModel): Unit = {
//    val batchKeys = new LongOpenHashSet(keys.size)
    val batchKeys = new LongOpenHashSet()
//    keys.foreach(k => batchKeys.add(k))
    val firstKeys = new LongOpenHashSet()

    val flags = model.readNodes(keys.clone())
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        // node with label
        batchKeys.add(keys(idx))
        var j = indptr(idx)
        while (j < indptr(idx + 1)) {
          val n = neighbors(j)
          if (!batchKeys.contains(n)) {
//            firstKeys.add(n)
            batchKeys.add(n)
          }
          if (!firstKeys.contains(n)) {
            firstKeys.add(n)
          }
          j += 1
        }
      }
    }

    // now batchKeys contain all the srcs we needed
    model.updateSrcKeys(batchKeys.toLongArray)

  }

  def egoNetworks(model: EgoNetworkPSModel): Iterator[(Long, Long)] ={
    val flags = model.readNodes(keys.clone())
    var edgesOut = new ArrayBuffer[(Long, Long)]
    for (idx <- keys.indices) {
      if (flags.get(keys(idx)) > 0) {
        var edges = new ArrayBuffer[(Long, Long)]
        edges.append((keys(idx) - indexGap, keys(idx) - indexGap))
        val neighbs = neighbors.slice(indptr(idx), indptr(idx+1))
        neighbs.foreach { nei => {
          if (nei < 0) {
            edges.append((nei * -1 - indexGap, keys(idx) - indexGap))
          } else {
            edges.append((keys(idx) - indexGap, nei - indexGap))
          }
        }
        }

        val secondNeighbs = model.sampleNeighbors(abs(neighbs), -1)
        val it = secondNeighbs.long2ObjectEntrySet().fastIterator()
        while (it.hasNext) {
          val entry = it.next()
          val ns = entry.getValue
          if (!ns.isEmpty) {
            ns.foreach { nei =>
              if (neighbs.contains(nei) || neighbs.contains(nei * -1)) {
                val proposal = if (nei < 0 ) {(nei * -1 - indexGap, entry.getLongKey - indexGap)} else {(entry.getLongKey - indexGap, nei - indexGap)}
                if (!edges.contains(proposal)) {edges.append(proposal)}
              }
            }
          }
        }
        edgesOut ++= edges
      }
    }
    edgesOut.iterator
  }

}

private[egonetwork]
object EgoNetworkPSPartition {
  def apply(index: Int, iterator: Iterator[(Long, Iterable[Long])], indexGap: Long): EgoNetworkPSPartition = {
    val indptr = new IntArrayList()
    val keys = new LongArrayList()
    val neighbors = new LongArrayList()

    indptr.add(0)
    while (iterator.hasNext) {
      val entry = iterator.next()
      val (node, ns) = (entry._1, entry._2)
      ns.foreach(n => neighbors.add(n))
      indptr.add(neighbors.size())
      keys.add(node)
    }

    new EgoNetworkPSPartition(index, keys.toLongArray,
      indptr.toIntArray, neighbors.toLongArray, indexGap)
  }
}
