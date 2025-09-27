#!/bin/bash

# Before run Spark on Angel application, you must follow the steps:
# 1. confirm Hadoop and Spark have ready in your environment
# 2. unzip angel-<version>-bin.zip to local directory
# 3. upload angel-<version>-bin directory to HDFS
# 4. set the following variables, SPARK_HOME, ANGEL_HOME, ANGEL_HDFS_HOME, ANGEL_VERSION


export HADOOP_HOME=<HADOOP_HOME>
export SPARK_HOME=<SPARK_HOME>
export ANGEL_HDFS_HOME=<ANGEL_HDFS_HOME>
export ANGEL_VERSION=3.3.0
export PYTORCH_ON_ANGEL_VERSION=0.4.0

scala_jar=scala-library-2.12.15.jar
netty_jar=netty-buffer-4.1.74.Final.jar,netty-common-4.1.74.Final.jar,netty-transport-4.1.74.Final.jar,netty-resolver-4.1.74.Final.jar
external_jar=httpclient-4.1.2.jar,httpcore-4.1.2.jar,chill_2.12-0.10.0.jar,chill-java-0.10.0.jar,algs4-1.0.3.jar,json4s-ast_2.12-3.6.12.jar,json4s-core_2.12-3.6.12.jar,json4s-jackson_2.12-3.6.12.jar,json4s-scalap_2.12-3.6.12.jar,fastutil-8.2.2.jar,htrace-core-2.05.jar,sizeof-0.3.0.jar,kryo-shaded-4.0.0.jar,minlog-1.3.0.jar,sketches-core-0.8.1.jar,memory-0.8.1.jar,commons-pool-1.6.jar,hll-1.6.0.jar,snappy-java-1.0.4.1.jar,flatbuffers-java-1.12.0.jar,groovy-all-2.4.7.jar,snakeyaml-1.16.jar 
angel_ps_jar=stream-2.9.6.jar,angel-ps-core-${ANGEL_VERSION}.jar,angel-ps-mllib-${ANGEL_VERSION}.jar,angel-ps-examples-${ANGEL_VERSION}.jar,angel-ps-psf-${ANGEL_VERSION}.jar,angel-ps-graph-${ANGEL_VERSION}.jar

sona_jar=spark-on-angel-core-${ANGEL_VERSION}.jar,spark-on-angel-mllib-${ANGEL_VERSION}.jar,spark-on-angel-graph-${ANGEL_VERSION}.jar,pytorch-on-angel-mllib-${PYTORCH_ON_ANGEL_VERSION}.jar
sona_psf_jar=spark-on-angel-mllib-${ANGEL_VERSION}-ps.jar,spark-on-angel-examples-${ANGEL_VERSION}-ps.jar,spark-on-angel-mllib-${ANGEL_VERSION}.jar,spark-on-angel-graph-${ANGEL_VERSION}.jar,pytorch-on-angel-mllib-${PYTORCH_ON_ANGEL_VERSION}.jar
sona_external_jar=commons-logging-1.1.1.jar,chill_2.12-0.10.0.jar,chill-java-0.10.0.jar,algs4-1.0.3.jar,fastutil-8.2.2.jar,htrace-core-2.05.jar,sizeof-0.3.0.jar,kryo-shaded-4.0.0.jar,minlog-1.3.0.jar,memory-0.8.1.jar,commons-pool-1.6.jar,hll-1.6.0.jar,json4s-jackson_2.12-3.6.12.jar,jniloader-1.1.jar,native_system-java-1.1.jar,arpack_combined_all-0.1.jar,core-1.1.2.jar,netlib-native_ref-linux-armhf-1.1-natives.jar,netlib-native_ref-linux-i686-1.1-natives.jar,netlib-native_ref-linux-x86_64-1.1-natives.jar,netlib-native_system-linux-armhf-1.1-natives.jar,netlib-native_system-linux-i686-1.1-natives.jar,netlib-native_system-linux-x86_64-1.1-natives.jar

dist_jar=${external_jar},${angel_ps_jar},${sona_psf_jar},${scala_jar},${netty_jar}
local_jar=${external_jar},${angel_ps_jar},${sona_jar},${sona_external_jar}


for f in `echo $dist_jar | awk -F, '{for(i=1; i<=NF; i++){ print $i}}'`; do
        jar=${ANGEL_HDFS_HOME}/lib/${f}
    if [ "$SONA_ANGEL_JARS" ]; then
        SONA_ANGEL_JARS=$SONA_ANGEL_JARS,$jar
    else
        SONA_ANGEL_JARS=$jar
    fi
done

export SONA_ANGEL_JARS

echo "=========================="
echo $SONA_ANGEL_JARS

for f in `echo $local_jar | awk -F, '{for(i=1; i<=NF; i++){ print $i}}'`; do
        jar=${ANGEL_HDFS_HOME}/lib/${f}
    if [ "$SONA_SPARK_JARS" ]; then
        SONA_SPARK_JARS=$SONA_SPARK_JARS,$jar
    else
        SONA_SPARK_JARS=$jar
    fi
done

echo SONA_SPARK_JARS: $SONA_SPARK_JARS

export SONA_SPARK_JARS
