#include "paddle_inference_api.h"
#include <vector>
#include <string>
#include <cstdio>
namespace paddle {
void createConfig(NativeConfig *config,const std::string& model_dirname) {
	config->use_gpu=false;
	config->SetCpuMathLibraryNumThreads(1);

	config->prog_file=model_dirname+"model";
	config->param_file=model_dirname+"param";

	config->specify_input_name=true;
}

void run(int batch_size,const std::string& model_dirname) {
	NativeConfig config;
	createConfig(&config,model_dirname);

	auto predictor=CreatePaddlePredictor(config);

	int c=3,h=300,w=300;
	float *data=new float[batch_size*c*w*h];
	FILE* f=fopen("mat.txt","r");
	for(int i=0;i<c*h*w;i++) fscanf(f,"%f",data+i);
	fclose(f);
	for(int i=0;i<50;i++) printf("%f\n",data[i]);
	PaddleTensor tensor;
	tensor.name="image";
	tensor.shape=std::vector<int>({batch_size,c,h,w});
	tensor.data=PaddleBuf(static_cast<void*>(data),sizeof(float)*(batch_size*c*h*w));
	tensor.dtype=PaddleDType::FLOAT32;
	std::vector<PaddleTensor> ptf(1,tensor);

	std::vector<PaddleTensor> output;
	predictor->Run(ptf,&output,batch_size);

	const size_t num_elements=output.front().data.length()/(sizeof(float));
	auto *data_out=static_cast<float *> (output.front().data.data());
	int cnt=0;
	for(int i=0;i<num_elements;i+=6) {
		//if(data[i+1]<0.6) continue;
		for(int j=0;j<6;j++) printf("%f ",data_out[i+j]);
		printf("\n");
	}
//	printf("%d\n",cnt);
}
}
int main() {
	paddle::run(1,"./cppmodel/");
}
