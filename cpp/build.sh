#!/bin/bash

OUT_DIR=./out

mkdir -p $OUT_DIR
rm -rf $OUT_DIR/*

cd $OUT_DIR
cmake ..
make
