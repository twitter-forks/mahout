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

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class InnerJoinFilterJob extends AbstractJob {

  public enum Counters {
    DROPPED_BY_FILTER,
    MISSING_FILTER,
    ACCEPTED_BY_FILTER,
    FILTERS
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new InnerJoinFilterJob(), args);
  }

  @Override public int run(String[] strings) throws Exception {
    addOutputOption();
    addOption("toBeFiltered", "tbf", "", true);
    addOption("filter", "f", "", true);
    addOption("filterDim", "fd", "", true);
    addOption("numReducers", "nr", "", "10");
    Map<String, List<String>> args = parseArguments(strings);
    if(args == null) {
      return -1;
    }
    Configuration conf = getConf();
    conf.set("filterDim", args.get(keyFor("filterDim")).get(0));
    Job job = new Job(conf, "");
    job.setNumReduceTasks(Integer.parseInt(args.get(keyFor("numReducers")).get(0)));
    FileInputFormat.addInputPath(job, new Path(args.get(keyFor("toBeFiltered")).get(0)));
    FileInputFormat.addInputPath(job, new Path(args.get(keyFor("filter")).get(0)));
    job.setInputFormatClass(SequenceFileInputFormat.class);
    FileOutputFormat.setOutputPath(job, getOutputPath());
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(VectorWritable.class);
    job.setMapperClass(Mapper.class);
    job.setReducerClass(FilterReducer.class);
    if(!job.waitForCompletion(true)) {
      return -1;
    }
    return 0;
  }

  public static class FilterReducer
      extends Reducer<IntWritable, VectorWritable, IntWritable, VectorWritable> {
    private int filterDim;
    @Override
    protected void setup(Context context) {
      filterDim = Integer.parseInt(context.getConfiguration().get("filterDim"));
    }

    @Override
    public void reduce(IntWritable key, Iterable<VectorWritable> vectors, Context ctx)
        throws IOException, InterruptedException {
      Vector filterVector = null;
      Vector toBeFilteredVector = null;
      for(VectorWritable vw : vectors) {
        if(vw.get().size() == filterDim) {
          filterVector = vw.get();
        } else {
          toBeFilteredVector = vw.get();
        }
      }
      if(filterVector != null) {
        ctx.getCounter(Counters.FILTERS).increment(1);
        if(toBeFilteredVector != null) {
          ctx.getCounter(Counters.ACCEPTED_BY_FILTER).increment(1);
          ctx.write(key, new VectorWritable(toBeFilteredVector));
        } else {
          ctx.getCounter(Counters.DROPPED_BY_FILTER).increment(1);
        }
      } else {
        ctx.getCounter(Counters.MISSING_FILTER).increment(1);
      }
    }
  }
}
