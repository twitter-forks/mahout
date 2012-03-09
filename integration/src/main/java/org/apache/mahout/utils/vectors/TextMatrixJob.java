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
package org.apache.mahout.utils.vectors;

import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirIterable;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

public class TextMatrixJob extends AbstractJob {
  private static final String DIM_OPT = "dimension";
  private static final String DENSE_OPT = "dense";
  private static final String DICT_OPT = "dictionaryPath";
  private static final String DICT_TYPE_OPT = "dictionaryType";
  private static final String SEPARATOR_OPT = "separator";
  private static final String NORM_OPT = "normalization";


  public enum Counters {
    VALID_VECTORS,
    VALID_VECTOR_ENTRIES,
    INVALID_VECTORS,
    EMPTY_VECTORS,
    INVALID_VECTOR_ENTRIES,
    OUT_OF_BOUNDS_VECTOR_ENTRIES
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new TextMatrixJob(), args);
  }

  @Override public int run(String[] strings) throws Exception {
    addInputOption();
    addOutputOption();
    addOption(DIM_OPT, "d", "Dimension of the output");
    addOption(DENSE_OPT, "dense", "Use DenseVector output");
    addOption(DICT_OPT, "dp", "Path to dictionary (SequenceFile<Text,IntWritable>)");
    addOption(DICT_TYPE_OPT, "dt", "Dictionary type: (text|sequencefile)", "sequencefile");
    addOption(SEPARATOR_OPT, "sep", "Separator char in text dictionary");
    addOption(NORM_OPT, "n", "Vector normalization for output (>= 0 or 'INF')");
    Map<String, List<String>> args = parseArguments(strings);
    if(args == null) {
      return -1;
    }
    Config config = new Config();
    config.setDictionaryPath(new Path(args.get(keyFor(DICT_OPT)).get(0)))
          .setDimension(args.containsKey(keyFor(DIM_OPT))
        ? Integer.parseInt(args.get(keyFor(DIM_OPT)).get(0)) : Integer.MAX_VALUE)
          .setDenseOutput(args.containsKey(keyFor(DENSE_OPT)))
          .setInputPath(getInputPath())
          .setOutputPath(getOutputPath())
          .setDictionaryType(args.get(keyFor(DICT_TYPE_OPT)).get(0))
          .setSeparator(args.get(keyFor(SEPARATOR_OPT)).get(0))
          .setNorm(args.get(keyFor(NORM_OPT)).get(0));
    return runJob(config);
  }

  private class Config {
    private Path inputPath;
    private Path outputPath;
    private Path dictionaryPath;
    private String dictionaryType;
    private int dimension;
    private boolean denseOutput;
    private String separator;
    private String norm;

    public String getNorm() {
      return norm;
    }

    public Config setNorm(String norm) {
      this.norm = norm;
      return this;
    }

    public Path getInputPath() {
      return inputPath;
    }

    public Config setInputPath(Path inputPath) {
      this.inputPath = inputPath;
      return this;
    }

    public Path getOutputPath() {
      return outputPath;
    }

    public Config setOutputPath(Path outputPath) {
      this.outputPath = outputPath;
      return this;
    }

    public Path getDictionaryPath() {
      return dictionaryPath;
    }

    public Config setDictionaryPath(Path dictionaryPath) {
      this.dictionaryPath = dictionaryPath;
      return this;
    }

    public String getDictionaryType() {
      return dictionaryType;
    }

    public Config setDictionaryType(String dictionaryType) {
      this.dictionaryType = dictionaryType;
      return this;
    }

    public int getDimension() {
      return dimension;
    }

    public Config setDimension(int dimension) {
      this.dimension = dimension;
      return this;
    }

    public boolean isDenseOutput() {
      return denseOutput;
    }

    public Config setDenseOutput(boolean denseOutput) {
      this.denseOutput = denseOutput;
      return this;
    }

    public String getSeparator() {
      return separator;
    }

    public Config setSeparator(String separator) {
      this.separator = separator;
      return this;
    }
  }
  public int runJob(Config config)
      throws ClassNotFoundException, IOException, InterruptedException {
    Configuration conf = getConf();
    conf.set(DICT_OPT, config.getDictionaryPath().toString());
    conf.setBoolean(DENSE_OPT, config.isDenseOutput());
    conf.set(DICT_TYPE_OPT, config.getDictionaryType());
    conf.setInt(DIM_OPT, config.getDimension());
    conf.set(SEPARATOR_OPT, config.getSeparator());
    conf.set(NORM_OPT, config.getNorm());
    Job job = new Job(conf, getClass().getName());
    FileInputFormat.addInputPath(job, config.getInputPath());
    FileOutputFormat.setOutputPath(job, config.getOutputPath());
    job.setMapperClass(TMMapper.class);
    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(TextMatrixJob.class);
    return job.waitForCompletion(true) ? 0 : -1;
  }

  public static class TMMapper extends Mapper<LongWritable, Text, IntWritable, VectorWritable> {
    private int dim = 0;
    private boolean denseOutput = false;
    private Pattern sepPattern = Pattern.compile("\t");
    private Map<String, Integer> termIdMap = Maps.newHashMap();
    private Double norm = null;
    @Override
    protected void setup(Context ctx) throws IOException {
      Configuration conf = ctx.getConfiguration();
      if(conf.get(SEPARATOR_OPT) != null) {
        sepPattern = Pattern.compile(conf.get(SEPARATOR_OPT));
      }
      if(conf.get(DICT_OPT) == null) {
        throw new IllegalStateException("No dictionary set!");
      }
      Path dictPath = new Path(conf.get(DICT_OPT));
      if(conf.get(DICT_TYPE_OPT, "sequencefile").equalsIgnoreCase("sequencefile")) {
        for(Pair<Text, IntWritable> pair
            : new SequenceFileDirIterable<Text, IntWritable>(dictPath, PathType.GLOB, conf)) {
          termIdMap.put(pair.getFirst().toString(), pair.getSecond().get());
        }
      } else {
        FSDataInputStream fis = FileSystem.get(conf).open(dictPath);
        BufferedReader br = new BufferedReader(new InputStreamReader(fis));
        String line = null;
        line = br.readLine();
        while(line != null) {
          String[] parts = sepPattern.split(line);
          if(parts.length > 1) {
            termIdMap.put(parts[0], Integer.parseInt(parts[parts.length - 1]));
          }
          line = br.readLine();
        }
      }
      if(conf.get(NORM_OPT) != null) {
        if(conf.get(NORM_OPT).equalsIgnoreCase("INF")) {
          norm = Double.POSITIVE_INFINITY;
        } else {
          norm = Double.parseDouble(conf.get(NORM_OPT));
          if(norm < 0) {
            throw new IllegalStateException("Negative norms not allowed");
          }
        }
      }
      denseOutput = conf.getBoolean(DENSE_OPT, false);
      dim = conf.getInt(DIM_OPT, Integer.MAX_VALUE);
      if(denseOutput && dim == Integer.MAX_VALUE) {
        throw new IllegalStateException("Dense output of cardinality:" + dim);
      }
    }

    @Override
    public void map(LongWritable offset, Text row, Context ctx)
        throws IOException, InterruptedException {
      String[] keyAndText = sepPattern.split(row.toString());
      if(keyAndText.length < 2) {
        ctx.getCounter(Counters.INVALID_VECTORS).increment(1);
        return;
      }
      try {
      // validate
        int key = Integer.parseInt(keyAndText[0]);
        Vector vector = denseOutput ? new DenseVector(dim) : new RandomAccessSparseVector(dim);
        String[] entries = keyAndText[1].split(",");
        int numValidVectorEntries = 0;
        for(String entry : entries) {
          try {
            String[] keyAndValue = entry.split(":");
            double value = Double.parseDouble(keyAndValue[1]);
            Integer termId = termIdMap.get(keyAndValue[0]);
            if(termId < dim) {
              vector.set(termId, value);
              numValidVectorEntries++;
              ctx.getCounter(Counters.VALID_VECTOR_ENTRIES).increment(1);
            } else {
              ctx.getCounter(Counters.OUT_OF_BOUNDS_VECTOR_ENTRIES).increment(1);
            }
          } catch (Exception e) {
            ctx.getCounter(Counters.INVALID_VECTOR_ENTRIES).increment(1);
          }
        }
        if(numValidVectorEntries > 0) {
          if(norm != null) {
            vector.assign(Functions.div(vector.norm(norm)));
          }
          ctx.write(new IntWritable(key), new VectorWritable(vector));
          ctx.getCounter(Counters.VALID_VECTORS).increment(1);
        } else {
          ctx.getCounter(Counters.EMPTY_VECTORS).increment(1);
        }
      } catch (NumberFormatException nfe) {
        ctx.getCounter(Counters.INVALID_VECTORS).increment(1);
      }
    }
  }
}
