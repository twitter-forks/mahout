package org.apache.mahout.clustering.lda.cvb;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.apache.mahout.clustering.ClusteringTestUtils;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.VectorFunction;
import org.junit.Test;

/**
 * Tests for {@link TopicModelUtils}.
 */
public class TestTopicModelUtils {
  public void assertColumnNormsEqualOrZero(Matrix expected, Matrix actual) {
    assertNotNull(actual);
    assertEquals(expected.columnSize(), actual.columnSize());
    for (int c = 0; c < expected.columnSize(); ++c) {
      Vector expectedColumn = expected.viewColumn(c);
      Vector actualColumn = actual.viewColumn(c);
      assertNotNull(actualColumn);
      double expectedNorm = expectedColumn.norm(1);
      double actualNorm = actualColumn.norm(1);
      if (actualNorm == 0) {
        continue;
      }
      assertEquals(expectedNorm, actualNorm, 1e-10);
    }
  }

  public long numNonzeros(Matrix matrix) {
    return (long) matrix.aggregateRows(new VectorFunction() {
      @Override
      public double apply(Vector v) {
        return v.getNumNondefaultElements();
      }
    }).norm(1);
  }

  public void assertFewerNonzeros(Matrix expected, Matrix actual) {
    assertTrue(numNonzeros(expected) > numNonzeros(actual));
  }

  @Test
  public void test() {
    double threshold = 0.5;
    Matrix topicTermCounts = ClusteringTestUtils.randomStructuredModel(20, 100);
    Matrix sparseTopicTermCounts = TopicModelUtils.sparsifyTopicTermCounts(
        topicTermCounts, threshold);
    assertColumnNormsEqualOrZero(topicTermCounts, sparseTopicTermCounts);
    assertFewerNonzeros(topicTermCounts, sparseTopicTermCounts);
  }
}
