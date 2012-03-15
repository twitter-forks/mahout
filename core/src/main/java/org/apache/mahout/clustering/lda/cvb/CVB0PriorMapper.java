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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.jobcontrol.Job;
import org.apache.hadoop.mapred.join.CompositeInputFormat;
import org.apache.hadoop.mapred.join.TupleWritable;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Random;

public class CVB0PriorMapper extends MapReduceBase implements
    Mapper<IntWritable, TupleWritable, IntWritable, VectorWritable> {
  private static final Logger log = LoggerFactory.getLogger(CVB0PriorMapper.class);

  public static final String DOCTOPIC_OUT = "doc.topic.output";
  private MultipleOutputs multipleOutputs;
  private Reporter reporter;
  private Random random;
  private float testFraction;
  private OutputCollector<IntWritable, VectorWritable> out;
  private int numTopics;

  private ModelTrainer modelTrainer;

  @Override
  public void configure(org.apache.hadoop.mapred.JobConf conf) {
    try {
    multipleOutputs = new MultipleOutputs(conf);
    CVBConfig c = new CVBConfig().read(conf);
    double eta = c.getEta();
    double alpha = c.getAlpha();
    long seed = c.getRandomSeed();
    random = RandomUtils.getRandom(seed);
    numTopics = c.getNumTopics();
    int numTerms = c.getNumTerms();
    int numUpdateThreads = c.getNumUpdateThreads();
    int numTrainThreads = c.getNumTrainThreads();
    double modelWeight = c.getModelWeight();
    testFraction = c.getTestFraction();
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

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void map(IntWritable docId, TupleWritable tuple,
                  OutputCollector<IntWritable, VectorWritable> out,
                  Reporter reporter) throws IOException {
    if(this.reporter == null || this.out == null) {
      this.reporter = reporter;
      this.out = out;
    }
    VectorWritable document = (VectorWritable) tuple.get(0);
    VectorWritable docTopicPrior = tuple.size() > 1
        ? (VectorWritable) tuple.get(1)
        : new VectorWritable(new DenseVector(numTopics).assign(1.0 / numTopics));

    TopicModel model = modelTrainer.getReadModel();
    Matrix docTopicModel = new SparseRowMatrix(numTopics, document.get().size(), true);
    // iterate one step on p(topic | doc)
    model.trainDocTopicModel(document.get(), docTopicPrior.get(), docTopicModel);
    // update the model
    model.update(docTopicModel);
    // emit the updated p(topic | doc)
    multipleOutputs.getCollector(DOCTOPIC_OUT, reporter).collect(docId, docTopicPrior);
  }

  @Override
  public void close() throws IOException {
    modelTrainer.stop();
    // emit the model
    for(MatrixSlice slice : modelTrainer.getReadModel()) {
      out.collect(new IntWritable(slice.index()),
          new VectorWritable(slice.vector()));
    }
    super.close();
  }

  public static void main(String[] args) throws IOException {
    JobConf conf = new JobConf();
    Job job = new Job(conf);

    MultipleOutputs.addNamedOutput(conf, DOCTOPIC_OUT,
        SequenceFileOutputFormat.class,
        IntWritable.class,
        VectorWritable.class);

    Path aPath = null;
    Path bPath = null;
    conf.setInputFormat(CompositeInputFormat.class);
    conf.set("mapred.join.expr", CompositeInputFormat.compose(
          "inner", SequenceFileInputFormat.class, aPath, bPath));
  }
}
