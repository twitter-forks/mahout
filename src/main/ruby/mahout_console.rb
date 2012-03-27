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

  def load_matrix(path)
    load_seqfile(path).inject({}) {|m, pair| m[pair.first.get] = pair.second.get; m }
  end

  def load_model(m)
    matrix = SparseRowMatrix.new(m.size, m.first[1].size)
    m.each{|k,v| matrix.assign_row(k, v)}
    TopicModel.new(matrix, 0.001, 0.001, nil, 1, 1.0)
  end

  def my_methods
    public_methods - Object.public_methods
  end
end

class Model
  include_package "org.apache.mahout.math"
  attr_reader :matrix, :dict, :label_dict, :model

  def initialize(fs, opts = {})
    @matrix = fs.load_matrix(opts[:model_path])
    @model = fs.load_model(matrix)
    @dict = fs.load_dict(opts[:dict_path])
    @label_dict = fs.load_dict(opts[:label_dict_path]) unless opts[:label_dict_path].nil?
  end

  def features_for(label, limit = nil)
    topic_id = @label_dict.nil? ? label.to_i : @label_dict.index(label)
    unless topic_id.nil? || @matrix[topic_id].nil?
      topic = @matrix[topic_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      topic_norm = topic.inject(0) {|sum,e| sum += e[1]}
      topic = topic[0..limit] unless limit.nil?
      topic.map{ |p| [@dict[p[0]], p[1]/topic_norm] }
    end
  end

  def topics_for(feature, limit = nil)
    feature_id = @dict.index(feature)
    unless feature_id.nil?
      topic_dist = @matrix.inject([]){|a,(topic_id,topic)| a << [@label_dict.nil? ? topic_id.to_s : label_dict[topic_id], topic.get(feature_id)]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = topic_dist.inject(0) {|sum,e| sum += e[1]}
      topic_dist = topic_dist[0..limit] unless limit.nil?
      topic_dist.map{ |p| [p[0], p[1] / norm] }
    end
  end

  def infer(vector, convergence = 0.0, max_iters = 20, prior = DenseVector.new(model.num_topics).assign(1.0 / model.num_topics))
    model.infer(vector, prior, convergence, max_iters)
  end
end

class UserDoc
  attr_reader :matrix, :dict, :model, :doc_topics, :label_dict

  def initialize(fs, model, opts = {})
    @matrix = fs.load_matrix(opts[:userdoc_path])
    @doc_topics = fs.load_matrix(opts[:doc_topic_path])
    @model = model
    @dict = model.dict
    @label_dict = model.label_dict
  end
  
  def features_for(user_id, limit = nil)
    unless user_id.nil? || matrix[user_id].nil?
      sorted = matrix[user_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = sorted.inject(0) {|sum,e| sum += e[1]}
      sorted = sorted[0..limit] unless limit.nil?
      sorted.map{ |p| [dict[p[0]], p[1]/norm] }
    end
  end

  def topics_for(user_id, limit = nil)
    unless user_id.nil? || doc_topics[user_id].nil?
      sorted = doc_topics[user_id].inject([]){|a,e| a << [e.index, e.get]; a}.sort{|a,b| b[1] <=> a[1] }
      norm = sorted.inject(0) {|sum,e| sum += e[1]}
      sorted = sorted[0..limit] unless limit.nil?
      sorted.map{ |p| [label_dict.nil? ? p[0] : label_dict[p[0]], p[1]/norm] }
    end
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
