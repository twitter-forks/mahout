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
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Counter;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixSlice;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;

public class PriorTrainingReducer extends MapReduceBase
    implements Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
  private static final Logger log = LoggerFactory.getLogger(PriorTrainingReducer.class);

  public enum Counters {
    DOCS,
    SKIPPED_DOC_IDS,
    UNUSED_PRIORS,
    USED_DOCS,
    DOCS_WITH_PRIORS
  }

  public static final String DOC_TOPICS = "docTopics";
  public static final String TOPIC_TERMS = "topicTerms";
  private ModelTrainer modelTrainer;
  private int maxIters;
  private int numTopics;
  private int numTerms;
  private boolean onlyLabeledDocs;
  private MultipleOutputs multipleOutputs;
  private Reporter reporter;

  protected ModelTrainer getModelTrainer() {
    return modelTrainer;
  }

  protected int getMaxIters() {
    return maxIters;
  }

  protected int getNumTopics() {
    return numTopics;
  }

  @Override
  public void configure(JobConf conf) {
    try {
      log.info("Retrieving configuration");
      multipleOutputs = new MultipleOutputs(conf);
      CVBConfig c = new CVBConfig().read(conf);
      double eta = c.getEta();
      double alpha = c.getAlpha();
      numTopics = c.getNumTopics();
      numTerms = c.getNumTerms();
      int numUpdateThreads = c.getNumUpdateThreads();
      int numTrainThreads = c.getNumTrainThreads();
      maxIters = c.getMaxItersPerDoc();
      double modelWeight = c.getModelWeight();
      onlyLabeledDocs = c.isUseOnlyLabeledDocs();

      log.info("Initializing read model");
      TopicModel readModel;
      Path[] modelPaths = CVB0Driver.getModelPaths(conf);
      if(modelPaths != null && modelPaths.length > 0) {
        readModel = new TopicModel(conf, eta, alpha, null, numUpdateThreads, modelWeight, modelPaths);
      } else {
        log.info("No model files found, starting with uniform p(term|topic) prior");
        Matrix m = new DenseMatrix(numTopics, numTerms);
        m.assign(1.0 / numTerms);
        readModel = new TopicModel(numTopics, numTerms, eta, alpha,
                                   RandomUtils.getRandom(c.getRandomSeed()), null,
                                   numTrainThreads, modelWeight);
      }

      log.info("Initializing write model");
      TopicModel writeModel = modelWeight == 1
          ? new TopicModel(new DenseMatrix(numTopics, numTerms),
                           new DenseVector(numTopics),
                           eta, alpha, null, numUpdateThreads, 1.0)
          : readModel;

      log.info("Initializing model trainer");
      modelTrainer = new ModelTrainer(readModel, writeModel, numTrainThreads, numTopics, numTerms);
      modelTrainer.start();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void reduce(IntWritable docId, Iterator<VectorWritable> vectors,
      OutputCollector<IntWritable,VectorWritable> out, Reporter reporter)
      throws IOException {
    if(this.reporter == null) {
      this.reporter = reporter;
    }
    Counter docCounter = reporter.getCounter(Counters.DOCS);
    docCounter.increment(1);
    Vector topicVector = null;
    Vector document = null;
    while(vectors.hasNext()) {
      VectorWritable v = vectors.next();
      /*
       *  NOTE: we are susceptible to the pathological case of numTerms == numTopics (which should
       *  never happen, as that would generate a horrible topic model), because we identify which
       *  vector is the "prior" and which is the document by document.size() == numTerms
       */
      if(v.get().size() == numTerms) {
        document = v.get();
      } else {
        topicVector = v.get();
      }
    }
    if(document == null) {
      if(topicVector != null) {
        reporter.getCounter(Counters.UNUSED_PRIORS).increment(1);
      }
      reporter.getCounter(Counters.SKIPPED_DOC_IDS).increment(1);
      return;
    } else if(topicVector == null && onlyLabeledDocs) {
      reporter.getCounter(Counters.SKIPPED_DOC_IDS).increment(1);
      return;
    } else {
      if(topicVector == null) {
        topicVector = new DenseVector(numTopics).assign(1.0 / numTopics);
      } else {
        if(reporter.getCounter(Counters.DOCS_WITH_PRIORS).getCounter() % 100 == 0) {
          long docsWithPriors = reporter.getCounter(Counters.DOCS_WITH_PRIORS).getCounter();
          long skippedDocs = reporter.getCounter(Counters.SKIPPED_DOC_IDS).getCounter();
          long total = reporter.getCounter(Counters.DOCS).getCounter();
          log.info("Processed {} docs total, {} with priors, skipped {} docs",
              new Object[]{total, docsWithPriors, skippedDocs});
        }
        reporter.getCounter(Counters.DOCS_WITH_PRIORS).increment(1);
      }
      modelTrainer.trainSync(document, topicVector, true, 1);
      multipleOutputs.getCollector(DOC_TOPICS, reporter)
                     .collect(docId, new VectorWritable(topicVector));
      reporter.getCounter(Counters.USED_DOCS).increment(1);
    }
  }

  @Override
  public void close() throws IOException {
    log.info("Stopping model trainer");
    modelTrainer.stop();

    log.info("Writing model");
    TopicModel model = modelTrainer.getReadModel();
    for(MatrixSlice topic : model) {
      multipleOutputs.getCollector(TOPIC_TERMS, reporter)
                     .collect(new IntWritable(topic.index()), new VectorWritable(topic.vector()));
    }
    multipleOutputs.close();
  }
}
