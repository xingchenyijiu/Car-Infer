#include "paddle_inference_api.h"

namespace paddle {
void CreateConfig(AnalysisConfig* config, const std::string& model_dirname) {
  // 模型从磁盘进行加载
  config->SetModel(model_dirname + "/model",                                                                                             
                      model_dirname + "/params");  
  // config->SetModel(model_dirname);
  // 如果模型从内存中加载，可以使用SetModelBuffer接口
  // config->SetModelBuffer(prog_buffer, prog_size, params_buffer, params_size); 
  config->EnableUseGpu(10 /*the initial size of the GPU memory pool in MB*/,  0 /*gpu_id*/);

  /* for cpu 
  config->DisableGpu();
  config->EnableMKLDNN();   // 可选
  config->SetCpuMathLibraryNumThreads(10);
  */

  // 当使用ZeroCopyTensor的时候，此处一定要设置为false。
  config->SwitchUseFeedFetchOps(false);
  // 当多输入的时候，此处一定要设置为true
  config->SwitchSpecifyInputNames(true);
  config->SwitchIrDebug(true); // 开关打开，会在每个图优化过程后生成dot文件，方便可视化。
  // config->SwitchIrOptim(false); // 默认为true。如果设置为false，关闭所有优化，执行过程同 NativePredictor
  // config->EnableMemoryOptim(); // 开启内存/显存复用
}

void RunAnalysis(int batch_size, std::string model_dirname) {
  // 1. 创建AnalysisConfig
  AnalysisConfig config;
  CreateConfig(&config, model_dirname);

  // 2. 根据config 创建predictor
  auto predictor = CreatePaddlePredictor(config);
  int channels = 3;
  int height = 224;
  int width = 224;
  float input[batch_size * channels * height * width] = {0};

  // 3. 创建输入
  // 同NativePredictor样例一样，此处可以使用PaddleTensor来创建输入
  // 以下的代码中使用了ZeroCopy的接口，同使用PaddleTensor不同的是：此接口可以避免预测中多余的cpu copy，提升预测性能。
  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputTensor(input_names[0]);
  input_t->Reshape({batch_size, channels, height, width});
  input_t->copy_from_cpu(input);

  // 4. 运行
 // CHECK(predictor->ZeroCopyRun());

  // 5. 获取输出
  std::vector<float> out_data;
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = 0;//std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());
}
}  // namespace paddle

int main() { 
  // 模型下载地址 http://paddle-inference-dist.cdn.bcebos.com/tensorrt_test/mobilenet.tar.gz
  paddle::RunAnalysis(1, "./mobilenet");
  return 0;
}

