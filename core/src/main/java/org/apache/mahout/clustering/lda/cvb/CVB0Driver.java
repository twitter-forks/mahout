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

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.common.mapreduce.VectorSumReducer;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.vectorizer.SparseVectorsFromSequenceFiles;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.List;

/**
 * See {@link CachingCVB0Mapper} for more details on scalability and room for improvement.
 * To try out this LDA implementation without using Hadoop, check out
 * {@link InMemoryCollapsedVariationalBayes0}.  If you want to do training directly in java code
 * with your own main(), then look to {@link ModelTrainer} and {@link TopicModel}.
 *
 * Usage: {@code ./bin/mahout cvb <i>options</i>}
 * <p>
 * Valid options include:
 * <dl>
 * <dt>{@code --input path}</td>
 * <dd>Input path for {@code SequenceFile<IntWritable, VectorWritable>} document vectors. See
 * {@link SparseVectorsFromSequenceFiles} for details on how to generate this input format.</dd>
 * <dt>{@code --dictionary path}</dt>
 * <dd>Path to dictionary file(s) generated during construction of input document vectors (glob
 * expression supported). If set, this data is scanned to determine an appropriate value for option
 * {@code --num_terms}.</dd>
 * <dt>{@code --output path}</dt>
 * <dd>Output path for topic-term distributions.</dd>
 * <dt>{@code --doc_topic_output path}</dt>
 * <dd>Output path for doc-topic distributions.</dd>
 * <dt>{@code --num_topics k}</dt>
 * <dd>Number of latent topics.</dd>
 * <dt>{@code --num_terms nt}</dt>
 * <dd>Number of unique features defined by input document vectors. If option {@code --dictionary}
 * is defined and this option is unspecified, term count is calculated from dictionary.</dd>
 * <dt>{@code --topic_model_temp_dir path}</dt>
 * <dd>Path in which to store model state after each iteration.</dd>
 * <dt>{@code --maxIter i}</dt>
 * <dd>Maximum number of iterations to perform. If this value is less than or equal to the number of
 * iteration states found beneath the path specified by option {@code --topic_model_temp_dir}, no
 * further iterations are performed. Instead, output topic-term and doc-topic distributions are
 * generated using data from the specified iteration.</dd>
 * <dt>{@code --max_doc_topic_iters i}</dt>
 * <dd>Maximum number of iterations per doc for p(topic|doc) learning. Defaults to {@code 10}.</dd>
 * <dt>{@code --doc_topic_smoothing a}</dt>
 * <dd>Smoothing for doc-topic distribution. Defaults to {@code 0.0001}.</dd>
 * <dt>{@code --term_topic_smoothing e}</dt>
 * <dd>Smoothing for topic-term distribution. Defaults to {@code 0.0001}.</dd>
 * <dt>{@code --random_seed seed}</dt>
 * <dd>Integer seed for random number generation.</dd>
 * <dt>{@code --test_set_percentage p}</dt>
 * <dd>Fraction of data to hold out for testing. Defaults to {@code 0.0}.</dd>
 * <dt>{@code --iteration_block_size block}</dt>
 * <dd>Number of iterations between perplexity checks. Defaults to {@code 10}. This option is
 * ignored unless option {@code --test_set_percentage} is greater than zero.</dd>
 * </dl>
 */
public class CVB0Driver extends AbstractJob {
  /**
   * Hadoop counters for various CVB0 jobs to aid in debugging.
   */
  public enum Counters {
    /**
     * Number of documents sampled from input set.
     *
     * @see CachingCVB0Mapper
     * @see CachingCVB0PerplexityMapper
     */
    SAMPLED_DOCUMENTS
  }

  private static final Logger log = LoggerFactory.getLogger(CVB0Driver.class);
  private static final String MODEL_PATHS = "mahout.lda.cvb.modelPath";

  @Override
  public int run(String[] args) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(CVBConfig.CONVERGENCE_DELTA_PARAM, "cd", "The convergence delta value", String.valueOf(CVBConfig.CONVERGENCE_DELTA_DEFAULT));
    addOption(DefaultOptionCreator.overwriteOption().create());

    addOption(CVBConfig.NUM_TOPICS_PARAM, "k", "Number of topics to learn", true);
    addOption(CVBConfig.NUM_TERMS_PARAM, "nt", "Vocabulary size", false);
    addOption(CVBConfig.DOC_TOPIC_SMOOTHING_PARAM, "a", "Smoothing for document/topic distribution", String.valueOf(CVBConfig.DOC_TOPIC_SMOOTHING_DEFAULT));
    addOption(CVBConfig.TERM_TOPIC_SMOOTHING_PARAM, "e", "Smoothing for topic/term distribution", String.valueOf(CVBConfig.TERM_TOPIC_SMOOTHING_DEFAULT));
    addOption(CVBConfig.DICTIONARY_PATH_PARAM, "dict", "Path to term-dictionary file(s) (glob expression supported)", false);
    addOption(CVBConfig.DOC_TOPIC_OUTPUT_PATH_PARAM, "dt", "Output path for the training doc/topic distribution", false);
    addOption(CVBConfig.MODEL_TEMP_PATH_PARAM, "mt", "Path to intermediate model path (useful for restarting)", false);
    addOption(CVBConfig.ITERATION_BLOCK_SIZE_PARAM, "block", "Number of iterations per perplexity check", String.valueOf(CVBConfig.ITERATION_BLOCK_SIZE_DEFAULT));
    addOption(CVBConfig.RANDOM_SEED_PARAM, "seed", "Random seed", String.valueOf(CVBConfig.RANDOM_SEED_DEFAULT));
    addOption(CVBConfig.TEST_SET_FRACTION_PARAM, "tf", "Fraction of data to hold out for testing", String.valueOf(CVBConfig.TEST_SET_FRACTION_DEFAULT));
    addOption(CVBConfig.NUM_TRAIN_THREADS_PARAM, "ntt", "number of threads per mapper to train with", String.valueOf(CVBConfig.NUM_TRAIN_THREADS_DEFAULT));
    addOption(CVBConfig.NUM_UPDATE_THREADS_PARAM, "nut", "number of threads per mapper to update the model with", String.valueOf(CVBConfig.NUM_UPDATE_THREADS_DEFAULT));
    addOption(CVBConfig.PERSIST_INTERMEDIATE_DOCTOPICS_PARAM, "pidt", "persist and update intermediate p(topic|doc)", "false");
    addOption(CVBConfig.DOC_TOPIC_PRIOR_PATH_PARAM, "dtp", "path to prior values of p(topic|doc) matrix");
    addOption(CVBConfig.MAX_ITERATIONS_PER_DOC_PARAM, "mipd", "max number of iterations per doc for p(topic|doc) learning", String.valueOf(CVBConfig.MAX_ITERATIONS_PER_DOC_DEFAULT));
    addOption(CVBConfig.MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM, "int", "max number of iterations per doc for p(topic|doc) inference", String.valueOf(CVBConfig.MAX_INFERENCE_ITERATIONS_PER_DOC_DEFAULT));
    addOption(CVBConfig.NUM_REDUCE_TASKS_PARAM, null, "number of reducers to use during model estimation", String.valueOf(CVBConfig.NUM_REDUCE_TASKS_DEFAULT));
    addOption(CVBConfig.ONLY_LABELED_DOCS_PARAM, "ol", "only use docs with non-null doc/topic priors", "false");
    addOption(buildOption(CVBConfig.BACKFILL_PERPLEXITY_PARAM, null, "enable back-filling of missing perplexity values", false, false, null));

    if(parseArguments(args) == null) {
      return -1;
    }

    int numTopics = Integer.parseInt(getOption(CVBConfig.NUM_TOPICS_PARAM));
    Path inputPath = getInputPath();
    Path topicModelOutputPath = getOutputPath();
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    int iterationBlockSize = Integer.parseInt(getOption(CVBConfig.ITERATION_BLOCK_SIZE_PARAM));
    float convergenceDelta = Float.parseFloat(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    float alpha = Float.parseFloat(getOption(CVBConfig.DOC_TOPIC_SMOOTHING_PARAM));
    float eta = Float.parseFloat(getOption(CVBConfig.TERM_TOPIC_SMOOTHING_PARAM));
    int numTrainThreads = Integer.parseInt(getOption(CVBConfig.NUM_TRAIN_THREADS_PARAM));
    int numUpdateThreads = Integer.parseInt(getOption(CVBConfig.NUM_UPDATE_THREADS_PARAM));
    int maxItersPerDoc = Integer.parseInt(getOption(CVBConfig.MAX_ITERATIONS_PER_DOC_PARAM));
    int maxInferenceItersPerDoc = Integer.parseInt(getOption(CVBConfig.MAX_INFERENCE_ITERATIONS_PER_DOC_PARAM));
    Path dictionaryPath = hasOption(CVBConfig.DICTIONARY_PATH_PARAM) ? new Path(getOption(CVBConfig.DICTIONARY_PATH_PARAM)) : null;
    int numTerms = hasOption(CVBConfig.NUM_TERMS_PARAM)
                 ? Integer.parseInt(getOption(CVBConfig.NUM_TERMS_PARAM))
                 : getNumTerms(getConf(), dictionaryPath);
    Path docTopicPriorPath = hasOption(CVBConfig.DOC_TOPIC_PRIOR_PATH_PARAM) ? new Path(getOption(CVBConfig.DOC_TOPIC_PRIOR_PATH_PARAM)) : null;
    boolean persistDocTopics = hasOption(CVBConfig.PERSIST_INTERMEDIATE_DOCTOPICS_PARAM)
            && getOption(CVBConfig.PERSIST_INTERMEDIATE_DOCTOPICS_PARAM).equalsIgnoreCase("true");
    Path docTopicOutputPath = hasOption(CVBConfig.DOC_TOPIC_OUTPUT_PATH_PARAM) ? new Path(getOption(CVBConfig.DOC_TOPIC_OUTPUT_PATH_PARAM)) : null;
    Path modelTempPath = hasOption(CVBConfig.MODEL_TEMP_PATH_PARAM)
                       ? new Path(getOption(CVBConfig.MODEL_TEMP_PATH_PARAM))
                       : getTempPath("topicModelState");
    long seed = hasOption(CVBConfig.RANDOM_SEED_PARAM)
              ? Long.parseLong(getOption(CVBConfig.RANDOM_SEED_PARAM))
              : System.nanoTime() % 10000;
    float testFraction = hasOption(CVBConfig.TEST_SET_FRACTION_PARAM)
                       ? Float.parseFloat(getOption(CVBConfig.TEST_SET_FRACTION_PARAM))
                       : 0.0f;
    int numReduceTasks = Integer.parseInt(getOption(CVBConfig.NUM_REDUCE_TASKS_PARAM));
    boolean backfillPerplexity = hasOption(CVBConfig.BACKFILL_PERPLEXITY_PARAM);
    boolean useOnlyLabeledDocs = hasOption(CVBConfig.ONLY_LABELED_DOCS_PARAM); // check!
    CVBConfig cvbConfig = new CVBConfig().setAlpha(alpha).setEta(eta)
        .setBackfillPerplexity(backfillPerplexity).setConvergenceDelta(convergenceDelta)
        .setDictionaryPath(dictionaryPath).setOutputPath(topicModelOutputPath)
        .setDocTopicOutputPath(docTopicOutputPath)
        .setDocTopicPriorPath(docTopicPriorPath).setInputPath(inputPath)
        .setPersistDocTopics(persistDocTopics)
        .setIterationBlockSize(iterationBlockSize).setMaxIterations(maxIterations)
        .setMaxItersPerDoc(maxItersPerDoc).setModelTempPath(modelTempPath)
        .setMaxInferenceItersPerDoc(maxInferenceItersPerDoc)
        .setNumReduceTasks(numReduceTasks).setNumTrainThreads(numTrainThreads)
        .setNumUpdateThreads(numUpdateThreads).setNumTerms(numTerms).setNumTopics(numTopics)
        .setTestFraction(testFraction).setRandomSeed(seed).setUseOnlyLabeledDocs(useOnlyLabeledDocs);
    return run(getConf(), cvbConfig);
  }

  private static int getNumTerms(Configuration conf, Path dictionaryPath) throws IOException {
    FileSystem fs = dictionaryPath.getFileSystem(conf);
    Text key = new Text();
    IntWritable value = new IntWritable();
    int maxTermId = -1;
    for (FileStatus stat : fs.globStatus(dictionaryPath)) {
      SequenceFile.Reader reader = new SequenceFile.Reader(fs, stat.getPath(), conf);
      while (reader.next(key, value)) {
        maxTermId = Math.max(maxTermId, value.get());
      }
    }
    return maxTermId + 1;
  }

   /**
   *
    * @param conf
    * @return
    * @throws ClassNotFoundException
   * @throws IOException
    * @throws InterruptedException
    */
   public static int run(Configuration conf, CVBConfig c)
       throws ClassNotFoundException, IOException, InterruptedException {
     // verify arguments
     Preconditions.checkArgument(c.getTestFraction() >= 0.0 && c.getTestFraction() <= 1.0,
        "Expected 'testFraction' value in range [0, 1] but found value '%s'", c.getTestFraction());
     Preconditions.checkArgument(!c.isBackfillPerplexity() || c.getTestFraction() > 0.0,
         "Expected 'testFraction' value in range (0, 1] but found value '%s'", c.getTestFraction());

      String infoString = "Will run Collapsed Variational Bayes (0th-derivative approximation) " +
        "learning for LDA on {} (numTerms: {}), finding {}-topics, with document/topic prior {}, " +
        "topic/term prior {}.  Maximum iterations to run will be {}, unless the change in " +
        "perplexity is less than {}.  Topic model output (p(term|topic) for each topic) will be " +
        "stored {}.  Random initialization seed is {}, holding out {} of the data for perplexity " +
       "check.  {}{}\n";
     log.info(infoString, new Object[] {c.getInputPath(), c.getNumTerms(), c.getNumTopics(),
         c.getAlpha(), c.getEta(), c.getMaxIterations(),
         c.getConvergenceDelta(), c.getOutputPath(), c.getRandomSeed(), c.getTestFraction(),
         c.isPersistDocTopics() ? "Persisting intermediate p(topic|doc)" : "",
         c.getDocTopicPriorPath() != null ? "  Using " + c.getDocTopicPriorPath()
                                            + " as p(topic|doc) prior" : ""});
     infoString = c.getDictionaryPath() == null
                ? "" : "Dictionary to be used located " + c.getDictionaryPath().toString() + '\n';
     infoString += c.getDocTopicOutputPath() == null
                ? "" : "p(topic|docId) will be stored " + c.getDocTopicOutputPath().toString() + '\n';
     log.info(infoString);

     // determine what the "current" iteration is (the iteration for which no data yet exists)
     int currentIteration = getCurrentIterationNumber(conf, c.getModelTempPath(), c.getMaxIterations());
     log.info("Current iteration number: {}", currentIteration);
     c.write(conf);

     // load (and optionally back-fill) perplexity values for previous iterations
     FileSystem fs = FileSystem.get(c.getModelTempPath().toUri(), conf);
     List<Double> perplexities = Lists.newArrayList();
     for (int i = 1; i < currentIteration; i++) {
       // form path to model
       Path modelPath = modelPath(c.getModelTempPath(), i);

       // read perplexity
       double perplexity = readPerplexity(conf, c.getModelTempPath(), i);
       if (Double.isNaN(perplexity)) {
         if (!(c.isBackfillPerplexity() && i % c.getIterationBlockSize() == 0)) {
           continue;
         }
         log.info("Backfilling perplexity at iteration {}", i);
         if (!fs.exists(modelPath)) {
           log.error("Model path '{}' does not exist; Skipping iteration {} perplexity calculation", modelPath.toString(), i);
           continue;
         }
         perplexity = calculatePerplexity(conf, c.getInputPath(), modelPath, i);
       }

       // register and log perplexity
       perplexities.add(perplexity);
       log.info("Perplexity at iteration {} = {}", i, perplexity);
     }

    long startTime = System.currentTimeMillis();
    int iterationCount = 0;
    boolean converged = false;
    for (; !converged && currentIteration <= c.getMaxIterations(); currentIteration++, iterationCount++) {
      // run current iteration
      log.info("About to run iteration {} of {}", currentIteration, c.getMaxIterations());
      runIteration(conf, c, currentIteration);

      if(c.getTestFraction() > 0 && currentIteration % c.getIterationBlockSize() == 0) {
        // calculate perplexity
        Path modelPath = modelPath(c.getModelTempPath(), currentIteration);
        double perplexity = calculatePerplexity(conf, c.getInputPath(), modelPath, currentIteration);
        perplexities.add(perplexity);
        double delta = rateOfChange(perplexities);
        log.info("Current perplexity = {}", perplexity);
        log.info("(p_curr - p_prev) / p_1 = {}; target = {}", delta, c.getConvergenceDelta());

        // test convergence
        if (c.getConvergenceDelta() > 0 && delta < c.getConvergenceDelta()) {
          log.info("Convergence achieved with perplexity {} and delta {}", perplexity, delta);
          converged = true;
        }
      }
    }
    long endTime = System.currentTimeMillis();
    log.info("Completed {} iterations in {} seconds", iterationCount, (endTime - startTime) / 1000);
    log.info("Perplexities: ({})", Joiner.on(", ").join(perplexities));

    // write final topic-term and doc-topic distributions
    int finalIteration = currentIteration - 1;
    Path finalIterationData = modelPath(c.getModelTempPath(), finalIteration);
    Job topicModelOutputJob = c.getOutputPath() != null
        ? writeTopicModel(conf, finalIterationData, c.getOutputPath())
        : null;
    Job docInferenceJob = c.getDocTopicOutputPath() != null
        ? writeDocTopicInference(conf, c.getInputPath(), finalIterationData, c.getDocTopicOutputPath())
        : null;
    if(topicModelOutputJob != null && !topicModelOutputJob.waitForCompletion(true)) {
      return -1;
    }
    if(docInferenceJob != null && !docInferenceJob.waitForCompletion(true)) {
      return -1;
    }
    return 0;
  }

  private static double rateOfChange(List<Double> perplexities) {
    int sz = perplexities.size();
    if(sz < 2) {
      return Double.MAX_VALUE;
    }
    return Math.abs(perplexities.get(sz - 1) - perplexities.get(sz - 2)) / perplexities.get(0);
  }

  private static double calculatePerplexity(Configuration conf, Path corpusPath, Path modelPath, int iteration)
      throws IOException,
      ClassNotFoundException, InterruptedException {
    String jobName = "Calculating perplexity for " + modelPath;
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CachingCVB0PerplexityMapper.class);
    job.setMapperClass(CachingCVB0PerplexityMapper.class);
    job.setCombinerClass(DualDoubleSumReducer.class);
    job.setReducerClass(DualDoubleSumReducer.class);
    job.setNumReduceTasks(1);
    job.setOutputKeyClass(DoubleWritable.class);
    job.setOutputValueClass(DoubleWritable.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, corpusPath);
    Path outputPath = perplexityPath(modelPath.getParent(), iteration);
    FileOutputFormat.setOutputPath(job, outputPath);
    setModelPaths(job.getConfiguration(), modelPath);
    HadoopUtil.delete(conf, outputPath);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException("Failed to calculate perplexity for: " + modelPath);
    }
    return readPerplexity(conf, modelPath.getParent(), iteration);
  }

  /**
   * Sums keys and values independently.
   */
  public static class DualDoubleSumReducer extends
    Reducer<DoubleWritable, DoubleWritable, DoubleWritable, DoubleWritable> {
    private final DoubleWritable outKey = new DoubleWritable();
    private final DoubleWritable outValue = new DoubleWritable();

    @Override
    public void run(Context context) throws IOException,
        InterruptedException {
      double keySum = 0.0;
      double valueSum = 0.0;
      while (context.nextKey()) {
        keySum += context.getCurrentKey().get();
        for (DoubleWritable value : context.getValues()) {
          valueSum += value.get();
        }
      }
      outKey.set(keySum);
      outValue.set(valueSum);
      context.write(outKey, outValue);
    }
  }

  /**
   * @param topicModelStateTemp
   * @param iteration
   * @return {@code double[2]} where first value is perplexity and second is model weight of those
   *         documents sampled during perplexity computation, or {@code null} if no perplexity data
   *         exists for the given iteration.
   * @throws IOException
   */
  public static double readPerplexity(Configuration conf, Path topicModelStateTemp, int iteration)
      throws IOException {
    FileSystem fs = FileSystem.get(topicModelStateTemp.toUri(), conf);
    Path perplexityPath = perplexityPath(topicModelStateTemp, iteration);
    if (!fs.exists(perplexityPath)) {
      log.warn("Perplexity path {} does not exist, returning NaN", perplexityPath);
      return Double.NaN;
    }
    double perplexity = 0;
    double modelWeight = 0;
    long n = 0;
    for (Pair<DoubleWritable, DoubleWritable> pair : new SequenceFileDirIterable<DoubleWritable, DoubleWritable>(
        perplexityPath, PathType.LIST, PathFilters.partFilter(), null, true, conf)) {
      modelWeight += pair.getFirst().get();
      perplexity += pair.getSecond().get();
      n++;
    }
    log.info("Read {} entries with total perplexity {} and model weight {}", new Object[] { n,
            perplexity, modelWeight });
    return perplexity / modelWeight;
  }

  private static Job writeTopicModel(Configuration conf, Path modelInput, Path output) throws IOException,
      InterruptedException, ClassNotFoundException {
    String jobName = String.format("Writing final topic/term distributions from %s to %s", modelInput,
        output);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CVB0Driver.class);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapperClass(CVB0TopicTermVectorNormalizerMapper.class);
    job.setNumReduceTasks(0);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileInputFormat.addInputPath(job, modelInput);
    FileOutputFormat.setOutputPath(job, output);
    job.submit();
    return job;
  }

  private static Job writeDocTopicInference(Configuration conf, Path corpus, Path modelInput, Path output)
      throws IOException, ClassNotFoundException, InterruptedException {
    String jobName = String.format("Writing final document/topic inference from %s to %s", corpus,
        output);
    log.info("About to run: " + jobName);
    Job job = new Job(conf, jobName);
    job.setMapperClass(CVB0DocInferenceMapper.class);
    job.setNumReduceTasks(0);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    FileSystem fs = FileSystem.get(modelInput.toUri(), conf);
    if (modelInput != null && fs.exists(modelInput)) {
      FileStatus[] statuses = fs.listStatus(modelInput, PathFilters.partFilter());
      URI[] modelUris = new URI[statuses.length];
      for(int i = 0; i < statuses.length; i++) {
        modelUris[i] = statuses[i].getPath().toUri();
      }
      DistributedCache.setCacheFiles(modelUris, conf);
    }
    FileInputFormat.addInputPath(job, corpus);
    FileOutputFormat.setOutputPath(job, output);
    job.setJarByClass(CVB0Driver.class);
    job.submit();
    return job;
  }

  public static Path modelPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "model-" + iterationNumber);
  }

  public static Path stage1OutputPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "tmp-" + iterationNumber);
  }

  public static Path perplexityPath(Path topicModelStateTempPath, int iterationNumber) {
    return new Path(topicModelStateTempPath, "perplexity-" + iterationNumber);
  }

  /**
   * Scans model state path for existing models, returning the number of the
   * iteration for which no data yet exists. For example, when no model data for
   * any iteration is found, this function will return 1.
   *
   * @param conf
   *          job configuration.
   * @param modelTempDir
   *          path to model state.
   * @param maxIterations
   *          max number of iterations to look for.
   * @return the one-based number of the "current" iteration. If no results
   *         exist, this function will return 1.
   * @throws IOException
   */
  private static int getCurrentIterationNumber(Configuration conf, Path modelTempDir, int maxIterations)
      throws IOException {
    log.info("Scanning path '{}' for existing data", modelTempDir);
    FileSystem fs = FileSystem.get(modelTempDir.toUri(), conf);
    int iterationNumber = 1;
    Path iterationPath = modelPath(modelTempDir, iterationNumber);
    while(fs.exists(iterationPath) && iterationNumber <= maxIterations) {
      log.info("Found previous state '{}'", iterationPath);
      iterationNumber++;
      iterationPath = modelPath(modelTempDir, iterationNumber);
    }
    return iterationNumber;
  }

  public static void runIteration(Configuration conf, CVBConfig c, int iterationNumber) throws IOException,
      ClassNotFoundException, InterruptedException {
    if(c.isPersistDocTopics() || c.getDocTopicPriorPath() != null) {
      runIterationWithDocTopicPriors(conf, c, iterationNumber);
    } else {
      runIterationNoPriors(conf, c, iterationNumber);
    }
  }

  /**
   *      IdMap[corpus, docTopics_i]
   *   ->
   *      PriorRed[corpus, docTopics_i; model_i]
   *   --multiOut-->
   *      model_frag_i+1
   *      docTopics_i+1
   *
   *     IdMap[model_frag_i+1]
   *   -->
   *     VectSumRed[model_frag_i+1]
   *   -->
   *     model_i+1
   *
   * @param conf the basic configuration
   * @param currentIteration
   * @throws IOException
   * @throws ClassNotFoundException
   * @throws InterruptedException
   */
  public static void runIterationWithDocTopicPriors(Configuration conf, CVBConfig c,
      final int currentIteration)
      throws IOException, ClassNotFoundException, InterruptedException {
    // define IO paths
    final int previousIteration = currentIteration - 1;
    final Path inputModelPath = modelPath(c.getModelTempPath(), previousIteration);
    final Path intermediateModelPath = getIntermediateModelPath(currentIteration, c.getModelTempPath());
    final Path intermediateTopicTermPath = getIntermediateTopicTermPath(currentIteration, c.getModelTempPath());
    final Path outputModelPath = modelPath(c.getModelTempPath(), currentIteration);
    Path docTopicInput;
    if(c.getDocTopicPriorPath() == null || (c.isPersistDocTopics() && currentIteration > 1)) {
      docTopicInput = getDocTopicPath(previousIteration, c.getModelTempPath());
    } else {
      docTopicInput = c.getDocTopicPriorPath();
    }

    // create and configure part 1 job
    String jobName1 = String.format("Part 1 of iteration %d of %d, input corpus %s, doc-topics: %s",
        currentIteration, c.getMaxIterations(), c.getInputPath(), docTopicInput);
    // TODO(Jake Mannix): is this really the ctor you want?
    JobConf jobConf1 = new JobConf(conf, CVB0Driver.class);
    jobConf1.setJobName(jobName1);
    jobConf1.setJarByClass(CVB0Driver.class);
    jobConf1.setMapperClass(Id.class);
    jobConf1.setClass("mapred.input.key.class", IntWritable.class, WritableComparable.class);
    jobConf1.setClass("mapred.input.value.class", VectorWritable.class, Writable.class);
    jobConf1.setMapOutputKeyClass(IntWritable.class);
    jobConf1.setMapOutputValueClass(VectorWritable.class);
    jobConf1.setReducerClass(PriorTrainingReducer.class);
    jobConf1.setNumReduceTasks(c.getNumReduceTasks());
    // configure job input
    jobConf1.setInputFormat(org.apache.hadoop.mapred.SequenceFileInputFormat.class);
    setModelPaths(jobConf1, inputModelPath);
    org.apache.hadoop.mapred.FileInputFormat.addInputPath(jobConf1, c.getInputPath());
    if(FileSystem.get(docTopicInput.toUri(), conf).globStatus(docTopicInput).length > 0) {
      org.apache.hadoop.mapred.FileInputFormat.addInputPath(jobConf1, docTopicInput);
    }
    // configure job output
    MultipleOutputs.addNamedOutput(jobConf1, PriorTrainingReducer.DOC_TOPICS,
        org.apache.hadoop.mapred.SequenceFileOutputFormat.class, IntWritable.class, VectorWritable.class);
    MultipleOutputs.addNamedOutput(jobConf1, PriorTrainingReducer.TOPIC_TERMS,
        org.apache.hadoop.mapred.SequenceFileOutputFormat.class, IntWritable.class, VectorWritable.class);
    org.apache.hadoop.mapred.FileOutputFormat.setOutputPath(jobConf1, intermediateModelPath);
    // remove any existing output and invoke part 1 job
    log.info("About to run: " + jobName1);
    HadoopUtil.delete(conf, intermediateModelPath);
    RunningJob runningJob = JobClient.runJob(jobConf1);
    if(!runningJob.isComplete()) {
      throw new InterruptedException(String.format("Failed to complete iteration %d stage 1",
          currentIteration));
    }

    // create and configure part 2 job
    String jobName2 = String.format("Part 2 of iteration %d of %d, input model fragments %s," +
        " output model state: %s", currentIteration, c.getMaxIterations(),
        intermediateTopicTermPath, outputModelPath);
    c.write(conf);
    Job job2 = new Job(conf, jobName2);
    job2.setJarByClass(CVB0Driver.class);
    job2.setMapperClass(Mapper.class);
    job2.setCombinerClass(VectorSumReducer.class);
    job2.setReducerClass(VectorSumReducer.class);
    job2.setNumReduceTasks(c.getNumReduceTasks());
    job2.setOutputKeyClass(IntWritable.class);
    job2.setOutputValueClass(VectorWritable.class);
    // configure job input
    job2.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(job2, intermediateTopicTermPath);
    // configure job output
    job2.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(job2, outputModelPath);
    // remove any existing output and invoke part 2 job
    log.info("About to run: " + jobName2);
    HadoopUtil.delete(conf, outputModelPath);
    if(!job2.waitForCompletion(true)) {
      throw new InterruptedException(String.format("Failed to complete iteration %d stage 2",
          currentIteration));
    }
  }

  public static void runIterationNoPriors(Configuration conf, CVBConfig c, int currentIteration)
      throws IOException, ClassNotFoundException, InterruptedException {
    final int previousIteration = currentIteration - 1;
    final Path modelInput = modelPath(c.getModelTempPath(), previousIteration);
    final Path modelOutput = modelPath(c.getModelTempPath(), currentIteration);
    String jobName = String.format("Iteration %d of %d, input model path: %s",
        currentIteration, c.getMaxIterations(), modelInput);
    Job job = new Job(conf, jobName);
    job.setJarByClass(CVB0Driver.class);
    job.setMapperClass(CachingCVB0Mapper.class);
    job.setCombinerClass(VectorSumReducer.class);
    job.setReducerClass(VectorSumReducer.class);
    job.setNumReduceTasks(c.getNumReduceTasks());
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    // configure job input
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileInputFormat.addInputPath(job, c.getInputPath());
    setModelPaths(job.getConfiguration(), modelInput);
    // configure job output
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    FileOutputFormat.setOutputPath(job, modelOutput);
    // remove existing output and invoke job
    log.info("About to run: " + jobName);
    HadoopUtil.delete(conf, modelOutput);
    if(!job.waitForCompletion(true)) {
      throw new InterruptedException(String.format("Failed to complete iteration %d stage 1",
          currentIteration));
    }
  }

  private static void setModelPaths(Configuration conf, Path modelPath) throws IOException {
    if (modelPath == null) {
      return;
    }
    FileSystem fs = FileSystem.get(modelPath.toUri(), conf);
    if (!fs.exists(modelPath)) {
      return;
    }
    FileStatus[] statuses = fs.listStatus(modelPath, PathFilters.partFilter());
    Preconditions.checkState(statuses.length > 0, "No part files found in model path '%s'", modelPath.toString());
    String[] modelPaths = new String[statuses.length];
    for (int i = 0; i < statuses.length; i++) {
      modelPaths[i] = statuses[i].getPath().toUri().toString();
    }
    conf.setStrings(MODEL_PATHS, modelPaths);
  }

  public static Path getIntermediateModelPath(int iterationNumber, Path topicModelStateTempPath) {
    return new Path(topicModelStateTempPath, "model-tmp-" + iterationNumber);
  }

  public static Path getDocTopicPath(int interationNumber, Path topicModelStateTempPath) {
    return new Path(getIntermediateModelPath(interationNumber, topicModelStateTempPath), "docTopics-*");
  }

  public static Path getIntermediateTopicTermPath(int iterationNumber, Path topicModelStateTempPath) {
    return new Path(getIntermediateModelPath(iterationNumber, topicModelStateTempPath), "topicTerms-*");
  }

  public static Path[] getModelPaths(Configuration conf) {
    String[] modelPathNames = conf.getStrings(MODEL_PATHS);
    if (modelPathNames == null || modelPathNames.length == 0) {
      return null;
    }
    Path[] modelPaths = new Path[modelPathNames.length];
    for (int i = 0; i < modelPathNames.length; i++) {
      modelPaths[i] = new Path(modelPathNames[i]);
    }
    return modelPaths;
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new Configuration(), new CVB0Driver(), args);
  }

  public static final class Id implements
      org.apache.hadoop.mapred.Mapper<IntWritable, VectorWritable, IntWritable, VectorWritable> {

    @Override public void map(IntWritable k, VectorWritable v,
        OutputCollector<IntWritable, VectorWritable> out, Reporter reporter) throws IOException {
      out.collect(k, v);
    }

    @Override public void close() throws IOException {
      // do nothing
    }

    @Override public void configure(JobConf jobConf) {
      // do nothing
    }
  }
}
