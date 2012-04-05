require 'java'
require 'irb'
require 'delegate'

module Kernel
  def suppress_warnings
    original_verbosity = $VERBOSE
    $VERBOSE = nil
    result = yield
    $VERBOSE = original_verbosity
    return result
  end
end

class Hdfs < SimpleDelegator
  include_package "org.apache.hadoop.fs"
  include_package "org.apache.hadoop.conf"
  include_package "org.apache.mahout.utils.vectors"
  include_package "org.apache.mahout.common.iterator"
  include_package "org.apache.hadoop.io"
  include_package "org.apache.mahout.common.iterator.sequencefile"
  include_package "org.apache.mahout.clustering.lda.cvb"
  include_package "org.apache.mahout.math"

  attr_reader :conf, :fs

  def initialize(fsname = ENV["HDFS_DEFAULT"])
    @conf = Configuration.new
    @conf.set("fs.default.name", fsname)
    @fs = FileSystem.get(@conf)
    super(@fs)
  end

  def ls(path)
    status = fs.list_status(Path.new(path))
    status.each{|s| puts s.path; s.path }
  end

  def open(path)
    fs.open(Path.new(path))
  end

  def file_lines(path)
    FileLineIterator.new(open(path))
  end

  def to_seqfile(in_path, out_path)
    text = file_lines(in_path)
    writer = SequenceFile.create_writer(fs, conf, Path.new(out_path), Text.java_class, IntWritable.java_class)
    text.each do |line|
      kv = line.split("\t")
      writer.append(Text.new(kv[0]), IntWritable.new(kv[1].to_i))
    end
    writer.close
  end

  def load_seqfile(path)
    SequenceFileDirIterable.new(Path.new(path), PathType::GLOB, nil, nil, true, conf)
  end

  def load_dict(path)
    load_seqfile(path).inject([]) {|arr, pair| arr[pair.second.get] = pair.first.to_s; arr }
  end

  def load_vector_hash(path)
    load_seqfile(path).inject({}) {|m, pair| m[pair.first.get] = pair.second.get; m }
  end

  def to_matrix(m)
    matrix = SparseRowMatrix.new(m.size, m.first[1].size)
    m.each{|k,v| matrix.assign_row(k, v) }
    matrix
  end

  def my_methods
    public_methods - Object.public_methods
  end

end

class Model
  include_package "org.apache.mahout.math"
  include_package "org.apache.mahout.clustering.lda.cvb"
  attr_reader :matrix_hash, :dict, :label_dict, :model, 
              :topic_feature_counts, :feature_freqs, :topic_freqs, :model_norm,
              :doc_hash, :doc_topics

  def self.help()
    puts "Instantiate via: Model.new(fs, {:model_path => _, :dict_path => _, :label_dict_path => _, :doc_path => _, :doc_topic_path => _})\n" +
         "where :model_path and :dict_path are required, but the rest are only needed if you want to have labeled topics or have documents\n" +
         "model.mahout_methods() will print out useful methods for this class";
  end
  
  def mahout_methods
    (public_methods - Object.public_methods).sort
  end

  def initialize(fs, opts = {})
    if (!opts[:model_path]) 
      raise ":model_path required to instantiate a model!"
    else
      puts "loading topic model path: #{opts[:model_path]}..." 
      @matrix_hash = fs.load_vector_hash(opts[:model_path])
      @topic_feature_counts = fs.to_matrix(matrix_hash)
      puts "loaded #{@topic_feature_counts.num_rows} topics with #{@topic_feature_counts.num_cols} features"
    end
    @model = TopicModel.new(topic_feature_counts, 0.001, 0.001, nil, 1, 1.0)
    if (!opts[:dict_path])
      raise ":dict_path required to instantiate a model!"
    else
      puts "loading dictionary path: #{opts[:dict_path]}..."
      @dict = fs.load_dict(opts[:dict_path])
      puts "loaded a #{@dict.size}-term dictionary"
    end
    if (!opts[:label_dict_path])
      puts "no :label_dict_path specified, all topics must be integer topic_ids"
    else
      puts "loading topic_id / label dictionary from #{opts[:label_dict_path]}..."
      @label_dict = fs.load_dict(opts[:label_dict_path]) 
      puts "loaded a #{@label_dict.size}-label dictionary"
    end
    if (!opts[:doc_path])
      puts "no :doc_path specified, will not load any document vectors"
    else
      puts "loading documents from #{opts[:doc_path]}..."
      @doc_hash = fs.load_vector_hash(opts[:doc_path])
      puts "loaded #{@doc_hash.size} documents"
    end
    if (!opts[:doc_topic_path])
      puts "no :doc_topic_path, will not load pre-inferred p(topic | doc_id) data"
    else
      puts "loading p(topic | doc_id) vectors from #{opts[:doc_topic_path]}..."
      @doc_topics = fs.to_matrix(fs.load_vector_hash(opts[:doc_topic_path]))
      puts "loaded #{@doc_topics.num_rows} doc-topic distributions"
    end
  end

  def features_for_topic(label, limit = nil)
    topic_id = label_dict ? label_dict.index(label) : label.to_i
    unless topic_id.nil? || matrix_hash[topic_id].nil?
      topic = matrix_hash[topic_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      topic_norm = topic.inject(0) {|sum,e| sum += e[1]}
      topic = topic[0..limit] unless limit.nil?
      topic.map{ |p| [dict[p[0]], p[1]/topic_norm] }
    end
  end

  def topics_for_feature(feature, limit = nil)
    feature_id = dict.index(feature)
    unless feature_id.nil?
      topic_dist = matrix_hash.inject([]){|a,(topic_id,topic)| a << [label_dict ? label_dict[topic_id] : topic_id.to_s, topic.get(feature_id)]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = topic_dist.inject(0) {|sum,e| sum += e[1]}
      topic_dist = topic_dist[0..limit] unless limit.nil?
      topic_dist.map{ |p| [p[0], p[1] / norm] }
    end
  end

  def features_for_doc(doc_id, limit = nil)
    if doc_id && doc_hash[doc_id]
      sorted = doc_hash[doc_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = sorted.inject(0) {|sum,e| sum += e[1]}
      sorted = sorted[0..limit] unless limit.nil?
      sorted.map{ |p| [dict[p[0]], p[1]/norm] }
    end
  end

  def topics_for_doc(doc_id, limit = nil)
    if doc_id && doc_topic_hash[doc_id]
      sorted = doc_topics[doc_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = sorted.inject(0) {|sum,e| sum += e[1]}
      sorted = sorted[0..limit] unless limit.nil?
      sorted.map{ |p| [label_dict.nil? ? p[0] : label_dict[p[0]], p[1]/norm] }
    end
  end

  def infer(s)
    if s.respond_to?(:split)
      tokens = s.split.map {|token| dict.index(token.downcase) }.compact
      v = org.apache.mahout.math.SequentialAccessSparseVector.new(dict.size, tokens.size)
      tokens.each {|t| v.set(t, 1.0) }
    else
      v = s
    end
    model.infer(v, 0, 100)
  end

  def related(s, i = 0)
    topic = top_k(infer(s), i+1)[0]
    topic = label_dict ? label_dict[topic] : topic
    features_for_topic(topic, 10).map{|a| a[0] }
  end

  def significant_topics(feature)
    feature_freqs ||= load_feature_freqs
    topic_freqs || load_topic_freqs
    fid = dict.index(feature)
    freq = feature_freqs[fid]
    (0..topic_feature_counts.num_rows).map {|tid| freq * topic_freq[tid] < topic_feature_counts.get(tid, fid) * model_norm ? (label_dict ? label_dict[fid] : fid) : nil }.compact
  end

  def feature_freqs
    @feature_freqs ||= (0..topic_feature_counts.num_columns).inject([]) {|a,f| a[f] = topic_feature_counts.view_column(f).norm(1); a}
    @model_norm ||= @feature_freqs.reduce(:+)
    @feature_freqs
  end

  def topic_freqs
    @topic_freqs ||= (0..topic_feature_counts.num_rows).inject([]) {|a,t| a[t] = topic_feature_counts.view_row(t).norm(1); a}
    @model_norm ||= @feature_freqs.reduce(:+)
    @topic_freqs
  end

  def top_k(v, k, dict = nil)
    sorted = v.inject({}) {|a,e| e.respond_to?(:get) ? a[e.index] = e.get : a[e[0]] = e[1] ; a}.sort {|a,b|  b[1] <=> a[1] }
    top = sorted[0..k]
    top.map {|k,v| dict ? dict[k] : k }
  end
end

module IRB
  def self.start(ap_path = nil)
    $0 = File::basename(ap_path, ".rb") if ap_path

    IRB.setup(ap_path)
    @CONF[:IRB_NAME] = 'mahout'
    @CONF[:AP_NAME] = 'mahout'

    @CONF[:PROMPT][:MAHOUT] = { # name of prompt mode
      :PROMPT_I => "mahout> ",  # normal prompt
      :PROMPT_S => "mahout* ",  # prompt for continuing strings
      :PROMPT_C => "mahout* ",  # prompt for continuing statement
      :RETURN => "    ==>%s\n"  # format to return value
    }

    @CONF[:PROMPT_MODE] = :MAHOUT

    @CONF[:BACK_TRACE_LIMIT] = 0 unless $fullBackTrace

    irb = Irb.new

    @CONF[:IRB_RC].call(irb.context) if @CONF[:IRB_RC]
    @CONF[:MAIN_CONTEXT] = irb.context

    trap("SIGINT") do
      irb.signal_handle
    end

    begin
      catch(:IRB_EXIT) do
        irb.eval_input
      end
    ensure
      irb_at_exit
    end
  end
end

# I don't know why IRB gives me lots of annoying "already initialized constant warnings", so I suppress them for now
suppress_warnings { IRB.start(__FILE__) }
