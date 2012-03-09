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

public class CVBConfig {
  // TODO: sensible defaults and/or checks for validity
  private Path inputPath;
  private Path outputPath;
  private int numTopics;
  private int numTerms;
  private double alpha;
  private double eta;
  private int maxIterations = 1;
  private int iterationBlockSize = 1;
  private double convergenceDelta;
  private Path dictionaryPath;
  private Path docTopicOutputPath;
  private Path modelTempPath;
  private Path docTopicPriorPath;
  private boolean persistDocTopics;
  private long seed = 1234;
  private double testFraction = 0;
  private int numTrainThreads = 1;
  private int numUpdateThreads = 1;
  private int maxItersPerDoc = 1;
  private int numReduceTasks = 1;
  private boolean backfillPerplexity;
  private boolean useOnlyLabeledDocs;

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

  public Path getOutputPath() {
    return outputPath;
  }

  public CVBConfig setOutputPath(Path outputPath) {
    this.outputPath = outputPath;
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

  public double getAlpha() {
    return alpha;
  }

  public CVBConfig setAlpha(double alpha) {
    this.alpha = alpha;
    return this;
  }

  public double getEta() {
    return eta;
  }

  public CVBConfig setEta(double eta) {
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

  public double getConvergenceDelta() {
    return convergenceDelta;
  }

  public CVBConfig setConvergenceDelta(double convergenceDelta) {
    this.convergenceDelta = convergenceDelta;
    return this;
  }

  public Path getDictionaryPath() {
    return dictionaryPath;
  }

  public CVBConfig setDictionaryPath(Path dictionaryPath) {
    this.dictionaryPath = dictionaryPath;
    return this;
  }

  public Path getDocTopicOutputPath() {
    return docTopicOutputPath;
  }

  public CVBConfig setDocTopicOutputPath(Path docTopicOutputPath) {
    this.docTopicOutputPath = docTopicOutputPath;
    return this;
  }

  public Path getModelTempPath() {
    return modelTempPath;
  }

  public CVBConfig setModelTempPath(Path modelTempPath) {
    this.modelTempPath = modelTempPath;
    return this;
  }

  public Path getDocTopicPriorPath() {
    return docTopicPriorPath;
  }

  public CVBConfig setDocTopicPriorPath(Path docTopicPriorPath) {
    this.docTopicPriorPath = docTopicPriorPath;
    return this;
  }

  public boolean isPersistDocTopics() {
    return persistDocTopics;
  }

  public CVBConfig setPersistDocTopics(boolean persistDocTopics) {
    this.persistDocTopics = persistDocTopics;
    return this;
  }

  public long getSeed() {
    return seed;
  }

  public CVBConfig setSeed(long seed) {
    this.seed = seed;
    return this;
  }

  public double getTestFraction() {
    return testFraction;
  }

  public CVBConfig setTestFraction(double testFraction) {
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
}
