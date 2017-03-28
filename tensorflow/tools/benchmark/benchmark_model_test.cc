/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <fstream>
#include "tensorflow/tools/benchmark/benchmark_model.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace {

// Status InitializeSession(int inter_op_threads, int intra_op_threads, const string& graph,
//                          std::unique_ptr<Session>* session,
//                          std::unique_ptr<StatSummarizer>* stats) {
//   LOG(INFO) << "Loading TensorFlow.";
// 
//   tensorflow::SessionOptions options;
//   tensorflow::ConfigProto& config = options.config;
//   if (inter_op_threads > 0) {
//     config.set_inter_op_parallelism_threads(inter_op_threads);
//   }
//   if (intra_op_threads > 0) {
//     config.set_intra_op_parallelism_threads(intra_op_threads);
//   }
//   LOG(INFO) << "Got config, " << config.device_count_size() << " devices";
// 
//   session->reset(tensorflow::NewSession(options));
//   tensorflow::GraphDef tensorflow_graph;
//   Status s = ReadBinaryProto(Env::Default(), graph, &tensorflow_graph);
//   if (!s.ok()) {
//     LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
//     return s;
//   }
// 
//   stats->reset(new tensorflow::StatSummarizer(tensorflow_graph));
// 
//   s = (*session)->Create(tensorflow_graph);
//   if (!s.ok()) {
//     LOG(ERROR) << "Could not create TensorFlow Session: " << s;
//     return s;
//   }
// 
//   // Clear the proto to save memory space.
//   tensorflow_graph.Clear();
//   return Status::OK();
// }

TEST(BenchmarkModelTest, InitializeAndRun) {
  const string dir = testing::TmpDir();
  const string filename_pb = io::JoinPath(dir, "graphdef.pb");

  // Create a simple graph and write it to filename_pb.
  const int input_width = 4000;
  const int input_height = 4000;
  benchmark_model::InputLayerInfo input1;
  input1.shape = TensorShape({input_width, input_height});
  input1.data_type = DT_FLOAT;
  benchmark_model::InputLayerInfo input2;
  input2.shape = TensorShape({input_width, input_height});
  input2.data_type = DT_FLOAT;
  const TensorShape constant_shape({input_height, input_width});

  Tensor constant_tensor1(DT_FLOAT, constant_shape);
  test::FillFn<float>(&constant_tensor1, [](int) -> float { return 3.0; });
  Tensor constant_tensor2(DT_FLOAT, constant_shape);
  test::FillFn<float>(&constant_tensor2, [](int) -> float { return 3.0; });

  auto root = Scope::NewRootScope().ExitOnError();
  auto placeholder1 =
      ops::Placeholder(root, DT_FLOAT, ops::Placeholder::Shape(input1.shape));
  auto placeholder2 =
      ops::Placeholder(root, DT_FLOAT, ops::Placeholder::Shape(input2.shape));
  input1.name = placeholder1.node()->name();
  input2.name = placeholder2.node()->name();
  auto m1 = ops::SparseMatMul(root, placeholder1, constant_tensor1);
  auto m2 = ops::SparseMatMul(root, placeholder2, constant_tensor2);
  auto m3 = ops::Add(root, m1, m2);
  const string output_name = m3.node()->name();

  GraphDef graph_def;
  TF_ASSERT_OK(root.ToGraphDef(&graph_def));
  string graph_def_serialized;
  graph_def.SerializeToString(&graph_def_serialized);
  TF_ASSERT_OK(
      WriteStringToFile(Env::Default(), filename_pb, graph_def_serialized));

  std::unique_ptr<Session> session;
  std::unique_ptr<StatSummarizer> stats;
  int inter, intra;
  std::ifstream file("./para.txt");
  if(file.is_open()) {
    string line;
    getline(file,line);
    std::stringstream ss(line);
    ss >> inter;
    ss >> intra;
  } else {
    std::cerr << "Cannot open para.txt\n";
  }
  file.close();
  tensorflow::benchmark_model::InitializeSession(inter, intra, filename_pb, &session, &stats);

  TF_ASSERT_OK(benchmark_model::TimeMultipleRuns(
      0.0, 1, {input1, input2}, {output_name}, session.get(), stats.get()));
}

}  // namespace
}  // namespace tensorflow
