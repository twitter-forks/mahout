package org.apache.mahout.clustering.lda.cvb;

import java.util.Iterator;
import java.util.List;
import java.util.PriorityQueue;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SparseRowMatrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.Vector.Element;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;

/**
 * Utilities for {@link TopicModel}s.
 */
public class TopicModelUtils {
  /**
   * Sparsifies input topicTermCounts matrix.
   *
   * @param topicTermCounts
   *          matrix containing topic-term counts to sparsify in-place.
   * @param threshold
   *          relative threshold on each topic's total term count. The term
   *          count elements for each topic are sorted by count desc and the top
   *          term count elements whose sum is less than the given threshold are
   *          preserved. Remaining term count elements are not set in output
   *          matrix. Columns of the output matrix are re-normalized after
   *          truncation process, assigning truncated term count mass evenly to
   *          those topics whose remaining term count is non-zero.
   * @return sparsified version of topicTermCounts.
   */
  public Matrix sparsifyTopicTermCounts(Matrix topicTermCounts, double threshold) {
    Preconditions.checkNotNull(topicTermCounts);
    Preconditions.checkArgument(0 < threshold,
        "Expected threshold > 0 but found %s", threshold);
    final int numTopics = topicTermCounts.rowSize();
    final int numTerms = topicTermCounts.columnSize();
    // storage for sparsified topic-term count vectors
    final Vector[] sparseTopicTermCounts = new Vector[numTopics];
    // storage for sums of truncated term counts
    final Vector truncatedTermCounts = new DenseVector(numTerms);
    // priority queue used to collect top-weighted vector entries
    final PriorityQueue<Entry> topTermCountEntries = new PriorityQueue<Entry>(
        (int) (numTerms * threshold));

    /*
     * Truncate topic-term vectors while keeping track of lost term count mass.
     * We use the lost count mass to perform term (column) normalization after
     * truncation.
     */

    // for each topic index
    for (int t = 0; t < numTopics; ++t) {
      // reset state
      topTermCountEntries.clear();

      // fetch term counts and iterator over non-zero elements
      final Vector termCounts = topicTermCounts.viewRow(t);
      final Iterator<Element> itr = termCounts.iterateNonZero();

      // determine term count threshold
      final double totalTermCount = termCounts.norm(1);
      final double termCountThreshold = totalTermCount * threshold;

      // iterate over non-zero term counts, keeping track of total term count
      double termCount = 0;
      while (itr.hasNext()) {
        Element e = itr.next();
        termCount += e.get();
        topTermCountEntries.add(new Entry(e.index(), e.get()));

        // remove elements with smallest count from queue till threshold is met
        while (termCount > termCountThreshold && !topTermCountEntries.isEmpty()) {
          Entry entry = topTermCountEntries.poll();
          int index = entry.getIndex();
          double count = entry.getValue();
          termCount -= count;
          // keep track of truncated mass for this term
          truncatedTermCounts.setQuick(index,
              truncatedTermCounts.getQuick(index) + count);
        }
      }

      // initialize output topic-term count vector
      Vector sparseTermCounts = new RandomAccessSparseVector(numTerms,
          topTermCountEntries.size());
      sparseTopicTermCounts[t] = sparseTermCounts;
      for (Entry e : topTermCountEntries) {
        sparseTermCounts.setQuick(e.getIndex(), e.getValue());
      }
    }

    /*
     * now iterate over terms, spreading each term's truncated mass evenly among
     * those topics for which the term still has non-zero count. To improve
     * feature-wise iteration efficiency, we keep track of current non-zero
     * iterator and element for each topic.
     */

    // non-zero topic-term count vector iterators
    final List<Iterator<Element>> topicTermElementIters = Lists
        .newArrayListWithCapacity(numTopics);
    // current non-zero topic-term count vector element for each topic
    final List<Element> topicTermElements = Lists
        .newArrayListWithCapacity(numTopics);
    // initialize topic iterators and elements
    for (int t = 0; t < numTopics; ++t) {
      Iterator<Element> itr = sparseTopicTermCounts[t].iterateNonZero();
      if (itr == null) {
        itr = Iterators.emptyIterator();
      }
      topicTermElementIters.add(itr);
      topicTermElements.add(itr.hasNext() ? itr.next() : null);
    }
    // current column of topic-term count elements
    final List<Element> nonzeroTopicElements = Lists
        .newArrayListWithCapacity(numTopics);

    // for each term index
    for (int f = 0; f < numTerms; ++f) {
      final double truncatedTermCount = truncatedTermCounts.get(f);
      if (truncatedTermCount == 0) {
        // no truncation occurred for this term; no normalization necessary
        continue;
      }

      // find topics for which current term has non-zero count
      nonzeroTopicElements.clear();
      for (int t = 0; t < numTopics; ++t) {
        Element e = topicTermElements.get(t);
        if (e == null) {
          continue;
        }
        final Iterator<Element> itr = topicTermElementIters.get(t);
        while (e != null && e.index() < f) {
          if (!itr.hasNext()) {
            e = null;
          } else {
            e = itr.next();
          }
        }
        topicTermElements.set(t, e);
        if (e == null || e.index() > f) {
          continue;
        }
        nonzeroTopicElements.add(e);
      }

      // deal with case where term has been removed from *all* topics
      if (nonzeroTopicElements.isEmpty()) {
        // TODO(Andy Schlaikjer): What should be done?
        continue;
      }

      // term count mass to add to each topic-term count
      final double termCountDelta = truncatedTermCount
          / nonzeroTopicElements.size();

      // update topic-term counts
      for (Element e : nonzeroTopicElements) {
        e.set(e.get() + termCountDelta);
      }
    }

    // create the sparse matrix
    return new SparseRowMatrix(numTopics, numTerms, sparseTopicTermCounts,
        true, true);
  }

  /**
   * Comparable struct for {@link Element} data. Natural ordering of
   * {@link Entry} instances is value desc, index asc.
   */
  private static final class Entry implements Comparable<Entry> {
    private final int index;
    private final double value;

    public Entry(int index, double value) {
      this.index = index;
      this.value = value;
    }

    public int getIndex() {
      return index;
    }

    public double getValue() {
      return value;
    }

    @Override
    public int compareTo(Entry o) {
      if (this == o) return 0;
      if (o == null) return 1;
      if (value > o.value) return -1;
      if (value < o.value) return 1;
      if (index < o.index) return -1;
      if (index > o.index) return 1;
      return 0;
    }
  }
}
