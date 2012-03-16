/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.lda.cvb;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.common.MemoryUtil;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class CachingCVB0PerplexityMapper extends
    Mapper<IntWritable, VectorWritable, DoubleWritable, DoubleWritable> {
  private static final Logger log = LoggerFactory.getLogger(CachingCVB0PerplexityMapper.class);
  private ModelTrainer modelTrainer;
  private int maxIters;
  private int numTopics;
  private float testFraction;
  private Random random;
  private Vector topicVector;
  private final DoubleWritable outKey = new DoubleWritable();
  private final DoubleWritable outValue = new DoubleWritable();

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    MemoryUtil.startMemoryLogger(5000);

    log.info("Retrieving configuration");
    Configuration conf = context.getConfiguration();
    CVBConfig config = new CVBConfig().read(conf);
    float eta = config.getEta();
    float alpha = config.getAlpha();
    long seed = config.getRandomSeed();
    random = RandomUtils.getRandom(seed);
    numTopics = config.getNumTopics();
    int numTerms = config.getNumTerms();
    int numUpdateThreads = config.getNumUpdateThreads();
    int numTrainThreads = config.getNumTrainThreads();
    maxIters = config.getMaxItersPerDoc();
    float modelWeight = config.getModelWeight();
    testFraction = config.getTestFraction();

    log.info("Initializing read model");
    TopicModel readModel;
    Path[] modelPaths = CVB0Driver.getModelPaths(conf);
    if(modelPaths != null && modelPaths.length > 0) {
      readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
    } else {
      log.info("No model files found");
      readModel = new TopicModel(numTopics, numTerms, eta, alpha, RandomUtils.getRandom(seed), null,
          numTrainThreads, modelWeight);
    }

    log.info("Initializing model trainer");
    modelTrainer = new ModelTrainer(readModel, null, numTrainThreads, numTopics, numTerms);

    log.info("Initializing topic vector");
    topicVector = new DenseVector(new double[numTopics]);
  }

  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
    MemoryUtil.stopMemoryLogger();
  }

  @Override
  public void map(IntWritable docId, VectorWritable document, Context context)
      throws IOException, InterruptedException{
    if (1 > testFraction && random.nextFloat() >= testFraction) {
      return;
    }
    context.getCounter(CVB0Driver.Counters.SAMPLED_DOCUMENTS).increment(1);
    outKey.set(document.get().norm(1));
    outValue.set(modelTrainer.calculatePerplexity(document.get(), topicVector.assign(1.0 / numTopics), maxIters));
    context.write(outKey, outValue);
  }
}
