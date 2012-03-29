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

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CVB0DocInferenceMapper extends CachingCVB0Mapper {
  private static final Logger log = LoggerFactory.getLogger(CVB0DocInferenceMapper.class);
  private double minRelPerplexityDiff;
  private int maxIterations;

  @Override
  protected void setup(Context context) throws IOException, InterruptedException {
    super.setup(context);
    minRelPerplexityDiff = config.getMinRelPreplexityDiff();
    maxIterations = config.getMaxInferenceItersPerDoc();
  }

  @Override
  public void map(IntWritable docId, VectorWritable doc, Context context)
      throws IOException, InterruptedException {
    Vector docTopics = modelTrainer.getReadModel().infer(doc.get(), minRelPerplexityDiff, maxIterations);
    context.write(docId, new VectorWritable(docTopics));
  }

  @Override
  protected void cleanup(Context context) {
    log.info("Stopping model trainer");
    modelTrainer.stop();
  }
}
