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
import org.apache.mahout.common.commandline.DefaultOptionCreator;

import com.google.common.base.Preconditions;

/**
 * CVB Configuration utilities. This class encapsulates all parameters required to invoke
 * {@link CVB0Driver}. Parameter names and default values are also defined here.
 */
public class CVBConfig {
  // parameter names
  public static final String INPUT_PATH_PARAM = DefaultOptionCreator.INPUT_OPTION;
  public static final String DICTIONARY_PATH_PARAM = "dictionary";
  public static final String DOC_TOPIC_PRIOR_PATH_PARAM = "doc_topic_prior_path";
  public static final String MODEL_TEMP_PATH_PARAM = "topic_model_temp_dir";
  public static final String OUTPUT_PATH_PARAM = DefaultOptionCreator.OUTPUT_OPTION;
  public static final String DOC_TOPIC_OUTPUT_PATH_PARAM = "doc_topic_output";
  public static final String NUM_TOPICS_PARAM = "num_topics";
  public static final String NUM_TERMS_PARAM = "num_terms";
  public static final String DOC_TOPIC_SMOOTHING_PARAM = "doc_topic_smoothing";
  public static final String TERM_TOPIC_SMOOTHING_PARAM = "term_topic_smoothing";
  public static final String MAX_ITERATIONS_PARAM = DefaultOptionCreator.MAX_ITERATIONS_OPTION;
  public static final String CONVERGENCE_DELTA_PARAM = DefaultOptionCreator.CONVERGENCE_DELTA_OPTION;
  public static final String ITERATION_BLOCK_SIZE_PARAM = "iteration_block_size";
  public static final String RANDOM_SEED_PARAM = "random_seed";
  public static final String TEST_SET_FRACTION_PARAM = "test_set_fraction";
  public static final String NUM_TRAIN_THREADS_PARAM = "num_train_threads";
  public static final String NUM_UPDATE_THREADS_PARAM = "num_update_threads";
  public static final String MAX_ITERATIONS_PER_DOC_PARAM = "max_doc_topic_iters";
  public static final String MODEL_WEIGHT_PARAM = "prev_iter_mult";
  public static final String NUM_REDUCE_TASKS_PARAM = "num_reduce_tasks";
  public static final String PERSIST_INTERMEDIATE_DOCTOPICS_PARAM = "persist_intermediate_doctopics";
  public static final String BACKFILL_PERPLEXITY_PARAM = "backfill_perplexity";
  public static final String ONLY_LABELED_DOCS_PARAM = "labeled_only";
  public static final String MIN_RELATIVE_PERPLEXITY_DIFF_PARAM = "min_rel_perplexity_diff";
  public static final String MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM = "max_inf_doc_topic_iters";

  // default values
  public static final float DOC_TOPIC_SMOOTHING_DEFAULT = 1e-5f;
  public static final float TERM_TOPIC_SMOOTHING_DEFAULT = 1e-5f;
  public static final int MAX_ITERATIONS_DEFAULT = 30;
  public static final int ITERATION_BLOCK_SIZE_DEFAULT = 5;
  public static final float CONVERGENCE_DELTA_DEFAULT = 1e-5f;
  public static final long RANDOM_SEED_DEFAULT = 1234l;
  public static final float TEST_SET_FRACTION_DEFAULT = 1e-2f;
  public static final int NUM_TRAIN_THREADS_DEFAULT = 4;
  public static final int NUM_UPDATE_THREADS_DEFAULT = 1;
  public static final int MAX_ITERATIONS_PER_DOC_DEFAULT = 10;
  public static final int NUM_REDUCE_TASKS_DEFAULT = 1;
  public static final float MODEL_WEIGHT_DEFAULT = 1f;
  public static final float MIN_RELATIVE_PERPLEXITY_DIFF_DEFAULT = 1e-8f;
  public static final int MAX_INFERENCE_ITERATIONS_PER_DOC_DEFAULT = 100;

  // TODO: sensible defaults and/or checks for validity
  private Path inputPath;
  private Path dictionaryPath;
  private Path docTopicPriorPath;
  private Path modelTempPath;
  private Path outputPath;
  private Path docTopicOutputPath;
  private int numTopics;
  private int numTerms;
  private float alpha = DOC_TOPIC_SMOOTHING_DEFAULT;
  private float eta = TERM_TOPIC_SMOOTHING_DEFAULT;
  private int maxIterations = MAX_ITERATIONS_DEFAULT;
  private int iterationBlockSize = ITERATION_BLOCK_SIZE_DEFAULT;
  private float convergenceDelta = CONVERGENCE_DELTA_DEFAULT;
  private boolean persistDocTopics;
  private long randomSeed = RANDOM_SEED_DEFAULT;
  private float testFraction = TEST_SET_FRACTION_DEFAULT;
  private int numTrainThreads = NUM_TRAIN_THREADS_DEFAULT;
  private int numUpdateThreads = NUM_UPDATE_THREADS_DEFAULT;
  private int maxItersPerDoc = MAX_ITERATIONS_PER_DOC_DEFAULT;
  private int numReduceTasks = NUM_REDUCE_TASKS_DEFAULT;
  private boolean backfillPerplexity;
  private boolean useOnlyLabeledDocs;
  private float modelWeight = MODEL_WEIGHT_DEFAULT;
  private float minRelPreplexityDiff = MIN_RELATIVE_PERPLEXITY_DIFF_DEFAULT;
  private int maxInferenceItersPerDoc = MAX_INFERENCE_ITERATIONS_PER_DOC_DEFAULT;

  public boolean isUseOnlyLabeledDocs() {
    return useOnlyLabeledDocs;
  }

  public CVBConfig setUseOnlyLabeledDocs(boolean useOnlyLabeledDocs) {
    this.useOnlyLabeledDocs = useOnlyLabeledDocs;
    return this;
  }

  public Path getInputPath() {
    return inputPath;
  }

  public CVBConfig setInputPath(Path inputPath) {
    this.inputPath = inputPath;
    return this;
  }

  public Path getDictionaryPath() {
    return dictionaryPath;
  }

  public CVBConfig setDictionaryPath(Path dictionaryPath) {
    this.dictionaryPath = dictionaryPath;
    return this;
  }

  public Path getDocTopicPriorPath() {
    return docTopicPriorPath;
  }

  public CVBConfig setDocTopicPriorPath(Path docTopicPriorPath) {
    this.docTopicPriorPath = docTopicPriorPath;
    return this;
  }

  public Path getModelTempPath() {
    return modelTempPath;
  }

  public CVBConfig setModelTempPath(Path modelTempPath) {
    this.modelTempPath = modelTempPath;
    return this;
  }

  public Path getOutputPath() {
    return outputPath;
  }

  public CVBConfig setOutputPath(Path outputPath) {
    this.outputPath = outputPath;
    return this;
  }

  public Path getDocTopicOutputPath() {
    return docTopicOutputPath;
  }

  /**
   * @param docTopicOutputPath path for doc-topic distributions. If set, once the model is trained a
   * final inference job will be executed to produce p(topic|doc) estimates for input documents.
   * Defaults to {@code null}.
   * @return this.
   */
  public CVBConfig setDocTopicOutputPath(Path docTopicOutputPath) {
    this.docTopicOutputPath = docTopicOutputPath;
    return this;
  }

  public int getNumTopics() {
    return numTopics;
  }

  public CVBConfig setNumTopics(int numTopics) {
    this.numTopics = numTopics;
    return this;
  }

  public int getNumTerms() {
    return numTerms;
  }

  public CVBConfig setNumTerms(int numTerms) {
    this.numTerms = numTerms;
    return this;
  }

  public float getAlpha() {
    return alpha;
  }

  public CVBConfig setAlpha(float alpha) {
    this.alpha = alpha;
    return this;
  }

  public float getEta() {
    return eta;
  }

  public CVBConfig setEta(float eta) {
    this.eta = eta;
    return this;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public CVBConfig setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }

  public int getIterationBlockSize() {
    return iterationBlockSize;
  }

  public CVBConfig setIterationBlockSize(int iterationBlockSize) {
    this.iterationBlockSize = iterationBlockSize;
    return this;
  }

  public float getConvergenceDelta() {
    return convergenceDelta;
  }

  public CVBConfig setConvergenceDelta(float convergenceDelta) {
    this.convergenceDelta = convergenceDelta;
    return this;
  }

  public boolean isPersistDocTopics() {
    return persistDocTopics;
  }

  public CVBConfig setPersistDocTopics(boolean persistDocTopics) {
    this.persistDocTopics = persistDocTopics;
    return this;
  }

  public long getRandomSeed() {
    return randomSeed;
  }

  public CVBConfig setRandomSeed(long randomSeed) {
    this.randomSeed = randomSeed;
    return this;
  }

  public float getTestFraction() {
    return testFraction;
  }

  public CVBConfig setTestFraction(float testFraction) {
    this.testFraction = testFraction;
    return this;
  }

  public int getNumTrainThreads() {
    return numTrainThreads;
  }

  public CVBConfig setNumTrainThreads(int numTrainThreads) {
    this.numTrainThreads = numTrainThreads;
    return this;
  }

  public int getNumUpdateThreads() {
    return numUpdateThreads;
  }

  public CVBConfig setNumUpdateThreads(int numUpdateThreads) {
    this.numUpdateThreads = numUpdateThreads;
    return this;
  }

  public int getMaxItersPerDoc() {
    return maxItersPerDoc;
  }

  public CVBConfig setMaxItersPerDoc(int maxItersPerDoc) {
    this.maxItersPerDoc = maxItersPerDoc;
    return this;
  }

  public int getNumReduceTasks() {
    return numReduceTasks;
  }

  public CVBConfig setNumReduceTasks(int numReduceTasks) {
    this.numReduceTasks = numReduceTasks;
    return this;
  }

  public boolean isBackfillPerplexity() {
    return backfillPerplexity;
  }

  public CVBConfig setBackfillPerplexity(boolean backfillPerplexity) {
    this.backfillPerplexity = backfillPerplexity;
    return this;
  }

  public float getModelWeight() {
    return modelWeight;
  }

  public CVBConfig setModelWeight(float modelWeight) {
    this.modelWeight = modelWeight;
    return this;
  }

  public float getMinRelPreplexityDiff() {
    return minRelPreplexityDiff;
  }

  public CVBConfig setMinRelPreplexityDiff(float minRelPreplexityDiff) {
    this.minRelPreplexityDiff = minRelPreplexityDiff;
    return this;
  }

  public int getMaxInferenceItersPerDoc() {
    return maxInferenceItersPerDoc;
  }

  public CVBConfig setMaxInferenceItersPerDoc(int maxInferenceItersPerDoc) {
    this.maxInferenceItersPerDoc = maxInferenceItersPerDoc;
    return this;
  }

  public void write(Configuration conf) {
    conf.setInt(NUM_TOPICS_PARAM, numTopics);
    conf.setInt(NUM_TERMS_PARAM, numTerms);
    conf.setFloat(DOC_TOPIC_SMOOTHING_PARAM, alpha);
    conf.setFloat(TERM_TOPIC_SMOOTHING_PARAM, eta);
    conf.setLong(RANDOM_SEED_PARAM, randomSeed);
    conf.setFloat(TEST_SET_FRACTION_PARAM, testFraction);
    conf.setInt(NUM_TRAIN_THREADS_PARAM, numTrainThreads);
    conf.setInt(NUM_UPDATE_THREADS_PARAM, numUpdateThreads);
    conf.setInt(MAX_ITERATIONS_PER_DOC_PARAM, maxItersPerDoc);
    conf.setFloat(MODEL_WEIGHT_PARAM, modelWeight);
    conf.setBoolean(ONLY_LABELED_DOCS_PARAM, useOnlyLabeledDocs);
    conf.setFloat(MIN_RELATIVE_PERPLEXITY_DIFF_PARAM, minRelPreplexityDiff);
    conf.setInt(MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM, maxInferenceItersPerDoc);
  }

  public CVBConfig read(Configuration conf) {
    setNumTopics(conf.getInt(NUM_TOPICS_PARAM, 0));
    setNumTerms(conf.getInt(NUM_TERMS_PARAM, 0));
    setAlpha(conf.getFloat(DOC_TOPIC_SMOOTHING_PARAM, 0));
    setEta(conf.getFloat(TERM_TOPIC_SMOOTHING_PARAM, 0));
    setRandomSeed(conf.getLong(RANDOM_SEED_PARAM, 0));
    setTestFraction(conf.getFloat(TEST_SET_FRACTION_PARAM, 0));
    setNumTrainThreads(conf.getInt(NUM_TRAIN_THREADS_PARAM, 0));
    setNumUpdateThreads(conf.getInt(NUM_UPDATE_THREADS_PARAM, 0));
    setMaxItersPerDoc(conf.getInt(MAX_ITERATIONS_PER_DOC_PARAM, 0));
    setModelWeight(conf.getFloat(MODEL_WEIGHT_PARAM, 0));
    setUseOnlyLabeledDocs(conf.getBoolean(ONLY_LABELED_DOCS_PARAM, false));
    setMinRelPreplexityDiff(conf.getFloat(MIN_RELATIVE_PERPLEXITY_DIFF_PARAM, -1));
    setMaxInferenceItersPerDoc(conf.getInt(MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM, 0));
    check();
    return this;
  }

  public void check() {
    checkPositive(NUM_TOPICS_PARAM, numTopics);
    checkPositive(NUM_TERMS_PARAM, numTerms);
    checkGreater(NUM_TERMS_PARAM, numTerms, numTopics);
    checkPositive(DOC_TOPIC_SMOOTHING_PARAM, alpha);
    checkPositive(TERM_TOPIC_SMOOTHING_PARAM, eta);
    checkPositive(RANDOM_SEED_PARAM, randomSeed);
    checkPositive(TEST_SET_FRACTION_PARAM, testFraction);
    checkPositive(NUM_TRAIN_THREADS_PARAM, numTrainThreads);
    checkPositive(NUM_UPDATE_THREADS_PARAM, numUpdateThreads);
    checkPositive(MAX_ITERATIONS_PER_DOC_PARAM, maxItersPerDoc);
    checkGreaterOrEqual(MODEL_WEIGHT_PARAM, modelWeight, 1);
    checkGreaterOrEqual(MIN_RELATIVE_PERPLEXITY_DIFF_PARAM, minRelPreplexityDiff, 0);
    checkPositive(MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM, maxInferenceItersPerDoc);
  }

  protected void checkGreater(String param, Number value, Number threshold) {
    Preconditions.checkArgument(value.doubleValue() > threshold.doubleValue(),
                                "Expecting %s > %d but found %s", param, threshold, value);
  }

  protected void checkGreaterOrEqual(String param, Number value, Number threshold) {
    Preconditions.checkArgument(value.doubleValue() >= threshold.doubleValue(),
                                "Expecting %s >= %d but found %s", param, threshold, value);
  }

  protected void checkPositive(String param, Number value) {
    checkGreater(param, value, 0);
  }
}
