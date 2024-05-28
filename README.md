wav2lip 是一个通过 声音和视频生成 口型的工具 找到demo.bat开启
服务启动：D:\AI\wav2lip\Wav2Lip-master\Wav2Lip-master\start.sh
服务启动：D:\AI\wav2lip\Wav2Lip-master\Wav2Lip-master\api.py

chatglm 相当于chatgpt的工具 找到 python api.py 或cli_demo.py 开启(  明天下午三点半 让张三 李四 王二麻子到第六会议室开会。上面一句话提取 开会时间 参会人员 参会地点)
 你是哪位?

--ChatGLM-Efficient-Tuning-main
   説明書.txt 里面有实验
   教程
   https://www.bilibili.com/video/BV12P411r7Mw/?spm_id_from=333.999.0.0&vd_source=529a62f1c90890479c94ff62b977c205
   #训练 在dataset_info.json中设置自己的如， jid2v2_cognition，然后建立一个文件jid2v2_cognition.json来设置自己的问题大
   set CUDA_VISIBLE_DEVICES=0
  python src/train_bash.py     --stage sft     --do_train     --model_name_or_path model/basemodel/chatglm2-6b     --dataset jid2v2_cognition      --dataset_dir data   --finetuning_type lora   --output_dir output/congnition   --overwrite_cache   --per_device_train_batch_size 2   --gradient_accumulation_steps 2    --lr_scheduler_type cosine    --logging_steps 10  --save_steps 1000   --learning_rate 1e-3   --num_train_epochs 10.0 --fp16 
  #test 测试
   set CUDA_VISIBLE_DEVICES=0
  python src/cli_demo.py  --model_name_or_path model/basemodel/chatglm2-6b  --checkpoint_dir  output/congnition
  #导出模型
  python src/export_model.py  --model_name_or_path model/basemodel/chatglm2-6b  --checkpoint_dir  output/congnition --output_dir output/path_to_save_model

--Langchain
   LLM的一个大语言模型 可以制作知识库 ，启动找到 python startup.py -a

whisper
模型下载地址：https://huggingface.co/ggerganov/whisper.cpp
根据输入的音频得到文字(模型位置在 C:\Users\Administrator\.cache\whisper) 执行test.py就可以实验
服务启动：D:\AI\whisper\whisper-main\whisper-main\start.sh
            新版本：    D:\AI\whisper\whisper3-main\whisper-main3\start.sh

whisper 微调训练
模型位置：C:\Users\Administrator\.cache\huggingface\hub\models--openai--whisper-tiny  C:\Users\Administrator\.cache\huggingface\hub\models--openai--whisper-large-v3 或者(models--models--whisper-large-v3)
直接使用原模型预测: python infer.py --audio_path=dataset/test.wav --model_path=openai/whisper-large-v3
D:\AI\whisper\Whisper-Finetune-master
训练:  1微调：python finetune.py --base_model=openai/whisper-tiny --output_dir=output/
         2合并：python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-best/ --output_dir=models/
         3评估：python evaluation.py --model_path=models/whisper-tiny-finetune --metric=cer
         4实测：python infer.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune 
山东潍坊实测
1微调：python finetune.py --base_model=openai/whisper-large-v3 --output_dir=output/
           python merge_lora.py --lora_model=output/whisper-medium/checkpoint-final/ --output_dir=models/
           python infer.py --audio_path=dataset/test2.wav --model_path=models/whisper-medium-finetune 




whisper windows版本
D:\AI\whisper\windowsform\WhisperDesktop
模型下载地址：https://huggingface.co/ggerganov/whisper.cpp/tree/main
小坑：如果碰到报错 Inspect the output of the command and see if you can locate CUDA libraries.
是因为 bitsandbytes之前不支持windows，可以执行以下动作
pip uninstall bitsandbytes
pip uninstall bitsandbytes-windows
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
坑2：报错      freeze_support() 这样标识的，说明不是用main函数启动的，在程序力用main函数启动


tts
模型自动下载地址  C:\Users\Administrator\.paddlespeech\models， D:\AI\tts\model ,D:\nltk_data，C:\ProgramData\miniconda3\envs\PaddleSpeech\Lib\site-packages\paddlespeech\t2s\models\fastspeech2，C:\Users\Administrator\.paddlenlp\models\bert-base-chinese\
模型地址       https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/docs/source/released_model.md#text-to-speech-models
教程 https://paddlespeech.readthedocs.io/en/latest/demo_video.html
测试：paddlespeech tts --input "是的，您需要带上房产证，户口本，以及来到我们营业大厅办理。" --output output.wav
服务启动：D:\AI\tts\tts\start.sh
小坑：如果报 np.complex,就改np.complex128  (C:\ProgramData\miniconda3\envs\PaddleSpeech\Lib\site-packages\librosa\core\constantq.py)
         在linux unbantu20.4下会有报错，（version `GLIBC_2.32' not found），这时候把 opencc版本降到1.1.0就好了 pip install OpenCC==1.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


bark
--bark-main
通过文字转语音(模型位置在 C:\Users\Administrator\.cache\suno\bark_v0,C:\Users\Administrator\.cache\torch\hub\checkpoints，大的C:\Users\Administrator\.cache\huggingface\hub\models--suno--bark,小的C:\Users\Administrator\.cache\huggingface\hub\models--suno--bark-small) 模型的测评总结:D:\AI\bark\bark-main\bark-main\bark\assets\prompts\speaker.md
不同的声音 D:\AI\bark\bark-main\bark-main\bark\assets\prompts
测试用例 D:\AI\bark\bark-main\bark-main\test.py
服务启动：D:\AI\bark\bark-main\bark-main\start.sh
--Bark-Voice-Cloning-main
  克隆人声给bark (https://www.bilibili.com/video/BV17w411X7xu/?spm_id_from=333.337.search-card.all.click&vd_source=529a62f1c90890479c94ff62b977c205)
----Bark-Coqui 应用程序
 启动使用python api.py  然后到http://127.0.0.1:7860/  模型下载到(Downloading model to C:\Users\Administrator\AppData\Local\tts)  (https://www.bilibili.com/video/BV17w411X7xu/?spm_id_from=333.337.search-card.all.click&vd_source=529a62f1c90890479c94ff62b977c205)
语音上传有些不好用，需要手工添加到examples目录下，然后在app.py下找到examples添加进去 

--Retrieval-based-Voice-Conversion-WebUI-main 克隆人声 页面 启动  根目录 python.exe gui_v1.py 和 python.exe infer-web.py   （https://www.bilibili.com/video/BV1pm4y1z7Gm/?vd_source=529a62f1c90890479c94ff62b977c205）
    测试步骤：
      到对应目录下，执行 python.exe infer-web.py 会弹出网页
      默认直接点一键训练
     然后到模型推理 选刷新银色列表和索引路径
     直接选转换，然后播放试试
      
     坑:
     1 分割人声用UVR5 ,目录在：D:\AI\bark\Retrieval-based-Voice-Conversion-WebUI-main\Retrieval-based-Voice-Conversion-WebUI-main\tools\UVR_v5.6.0_setup.exe
     2 中间某些whl需要编译，需要windows C++ 目录在：D:\AI\bark\Retrieval-based-Voice-Conversion-WebUI-main\Retrieval-based-Voice-Conversion-WebUI-main\tools\vs_BuildTools.exe

SD
stable diffusion   AI 作画 点击 ：A启动器.exe 来启动

embedding
m3e
D:\AI\chatglm\ChatGLM2-6B\LangChain\Langchain-Chatchat-master\Langchain-Chatchat-master\embeddings\__init__.py
或者
D:\AI\m3e (里面包含 中文的transformer bert)
服务启动：D:\AI\m3e\start.sh 

paddleocr 文字识别
 启动测试D:\AI\ocr\PaddleOCR\PaddleOCR-release-2.7\test.py 模型地址 C:\Users\Administrator/.paddleocr/whl\det\ch\ch_PP-OCRv4_det_infer\
服务：D:\AI\ocr\PaddleOCR\PaddleOCR-release-2.7\api.py

labelimg  给YOLO训练集打标签(王者荣耀)
-----vocdevkit-master 从VOC转 YOLO的工具
-----VOCdevkit  王者荣耀数据集
-----windows_v1.8.1 应用工具
-----yolov5-master 训练工具
整个流程： D:\AI\labelimg\说明.docx
先把图片进行(windows_v1.8.1 )应用工具进行贴标签，越多越好。然后用vocdevkit-master这个工具把VOC转 YOLO的格式，然后放到yolov5-master的VOCdevkit  下进行训练
增加King 的配置 D:\AI\labelimg\yolov5-master\yolov5-master\data\King.yaml和D:\AI\labelimg\yolov5-master\yolov5-master\models\King.yaml
启动：
训练模型-D:\AI\labelimg\yolov5-master\yolov5-master\train.py
测试看效果-D:\AI\labelimg\yolov5-master\yolov5-master\detect.py


Yolo5
https://github.com/HumanSignal/labelImg
YOLO模型:https://github.com/ultralytics/yolov5/releases
YOLO模型下载，已经写过的: https://gitee.com/la-la-pi/yolov5/blob/King/data/King.yaml
学习资料：
https://zhuanlan.zhihu.com/p/664082477
https://blog.csdn.net/shopkeeper_/article/details/124407230  https://www.bilibili.com/video/BV1f44y187Xg/?p=3&vd_source=529a62f1c90890479c94ff62b977c205



paddledetection 目标检测
文档 D:\AI\detection\PaddleDetection-release-2.6\PaddleDetection-release-2.6\docs\tutorials\QUICK_STARTED_cn.md
  
测试:
GPU 版本
set CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
非GPU

python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=false weights=model/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
模型地址C:\Users\Administrator\.cache\paddle\weights\ppyolo_r50vd_dcn_1x_coco.pdparams

指针识别
文档：https://aistudio.baidu.com/bd-cpu-02/user/592909/7112857/notebooks/7112857.ipynb 或者 D:\AI\detection\Paddlex\readme.ipynb
训练:D:\AI\detection\Paddlex\train_detection.py
模型：D:\AI\detection\Paddlex\output\meter_det\pretrain\DarkNet53_ImageNet1k_pretrained.tar
GPU：python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
运行（有错误但是不影响结果）：python reader_infer.py --detector_dir output/meter_det/best_model --segmenter_dir output/meter_seg/best_model --image meter_det/test/20190822_160.jpg --save_dir ./output --use_erode
API服务(有错误但是不影响结果)：D:\AI\detection\Paddlex\api.py


安全帽检测：
D:\AI\detection\safetyhelmet
yolo_train 训练
yolo_test 测试
教程
API 服务(返回数量)：D:\AI\detection\safetyhelmet\api.py



facenet 面部识别
先执行：D:\AI\facenet\facenet-retinaface-pytorch\facenet-retinaface-pytorch-main\facenet-retinaface-pytorch-main\encoding.py
然后执行预测:D:\AI\facenet\facenet-retinaface-pytorch\facenet-retinaface-pytorch-main\facenet-retinaface-pytorch-main\predict.py
输入对应的图片地址：Input image filename:D:\AI\facenet\facenet-retinaface-pytorch\facenet-retinaface-pytorch-main\facenet-retinaface-pytorch-main\img\obama.jpg

启动服务：D:\AI\facenet\facenet-retinaface-pytorch\facenet-retinaface-pytorch-main\facenet-retinaface-pytorch-main\api.py



Transformer
D:\AI\chatglm\ChatGLM3-6B\ChatGLM3-main\ChatGLM3-main\transformer
6.使用管道函数
英文(情感分析，电影数据集，阅读理解,完形填空,文本生成,*命名文本识别*,文本总结，翻译-英语到德语，特征提取-向量)
中文(可以对 评价结果1或0做分类，然后提出特征，然后通过2分类 进行算出 评价好与坏，用好了后，用起来的话，就是可以一句话能算出评价)
7.实战任务1中文分类 对照英文的情感分析
8.实战任务2中文填空 对照英文的完形填空
9实战任务3中文句子关系推断 对照英文的 文本生成

 https://github.com/lansinuote/Huggingface_Toturials/blob/main/7.%E4%B8%AD%E6%96%87%E5%88%86%E7%B1%BB.ipynb
模型地址 C:\Users\Administrator\.cache\huggingface\hub
模型网址https://huggingface.co/transformers/v2.9.1/_modules/transformers/pipelines.html
google资料：https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb


ollama 本地使用llama3
ollama run llama3
docker run -d --restart unless-stopped  --name ollama-webui   -p 11433:8080       -e OLLAMA_API_BASE_URL=http://192.168.31.160:11434/api -e WEBUI_SECRET_KEY=TkjGEiQ@5K^j   ghcr.io/open-webui/open-webui:main 
---------------------------------------------------------
docker
直接删除
docker run --rm --gpus=all nvidia/cuda:12.3.0-cudnn8-devel-ubuntu20.04 nvidia-smi
保留
docker run  --gpus=all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 nvidia-smi
保持状态
docker run -it  --gpus=all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 /bin/bash

执行以下命令更新源：
sudo apt-get update
执行以下命令安装 Python3 的一些依赖库：
sudo apt-get install libqgispython3.10.4
sudo apt-get install libpython3.10-stdlib
————————————————
版权声明：本文为CSDN博主「Z.Q.Feng」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_46584887/article/details/120701003


GPU使用
先把 torch cup版本删除
pip install D:\AI\chatglm\ChatGLM2-6B\ChatGLM2-6B-main\venv\wheel\torch-2.1.0+cu121-cp310-cp310-win_amd64.whl

加速
pip install  -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install -r requirements.txt -i https://mirror.baidu.com/pypi/simple

cuda版本 torch ，一般计算必须用  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip uninstall torch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision  --index-url https://download.pytorch.org/whl/cu121
pip install D:\AI\chatglm\ChatGLM2-6B\ChatGLM2-6B-main\venv\wheel\torch-2.1.0+cu121-cp310-cp310-win_amd64.whl

镜像替换站：
https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/
https://aliendao.cn/models/lj1995/VoiceConversionWebUI

huggingface的镜像替换 
export HF_ENDPOINT=https://hf-mirror.com


转发代理
fspcdiy.m184.6266668.com
47.90.37.184
fspcdiy
9HDJFgIp6ON9
 

python infer.py --audio_path=D:/Prod/客服/山东澭坊/20220916084920-1663289341.1088-13605368155-107-Inbound.wav --model_path=models/whisper-tiny-finetune 
----------------------------------------------------------------------

进入容器方法
docker exec -it 8769a4706374781256856091d82795f15a57e3e3c97e56dc322c23c6a3fccb5c /bin/bash
语音识别+语义识别 位置 python 3.10.4
/usr/src/pythonSample1 

语音识别 python websocketapi.py
语义识别 uvicorn mini:app --reload --host 0.0.0.0 --port 4004

tts Python 3.9.18
/usr/src/pythonSample2
conda create -n tts python=3.9.18
conda activate tts


 conda使用方法
 
#创建虚拟环境
conda create -n helloai python=3.10.4（3.6、3.7等）


# To activate this environment, use
#     使用
#     $ conda activate helloai
#
# To deactivate an active environment, use
#
#     $ conda deactivate
 
#激活虚拟环境
source activate helloai (虚拟环境名称)
 
#退出虚拟环境
source deactivate helloai (虚拟环境名称)
 
#删除虚拟环境
conda remove -n helloai (虚拟环境名称) --all
 conda remove -n helloai   --all

#查看安装了哪些包
conda list
 
#安装包
conda install package_name(包名)
conda install scrapy==1.3 # 安装指定版本的包
conda install -n 环境名 包名 # 在conda指定的某个环境中安装包
 
#查看当前存在哪些虚拟环境
conda env list 
#或 
conda info -e
#或
conda info --envs
 
#检查更新当前conda
conda update conda
 
#更新anaconda
conda update anaconda
 
#更新所有库
conda update --all
 
#更新python
conda update python
 
 
#其他举例
pip install torch
pip install torchvision
----------------------------------------------------
产品 最小版 cuda:1.2.1 包含了 语音 向量 tts
docker run -it -p 8765:8765 -p 4004:4004  -p 4005:4005 --gpus=all cuda:1.2.1 /bin/bash
conda activate tts
uvicorn api:app --reload --host 0.0.0.0 --port 4005
conda activate helloai
uvicorn mini:app --reload --host 0.0.0.0 --port 4004

[program:ai-root-service]
command=/bin/bash -c /usr/src/pythonSample1/ai.sh
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/ai_service.err.log
stdout_logfile=/var/log/ai_service.out.log

[program:asr-root-service]
command=/bin/bash -c  conda run -n helloai python /usr/src/pythonSample1/asr.py
autostart=true
autorestart=true
startretries=3
stderr_logfile=/var/log/asr_service.err.log
stdout_logfile=/var/log/asr_service.out.log

打包
pip install nuitka 
windows上命令
nuitka --standalone websocketapi.py
linux上命令
nuitka3 --standalone websocketapi.py

以这样形式打包
nuitka3 --standalone --module websocketapi.py
会生成文件：websocketapi.cpython-310-x86_64-linux-gnu.so
另外同一级目录直接使用就好了(asr.py)
import websocketapi
if __name__ == "__main__":
    websocketapi.runmain()

例子
执行：conda run -n helloai python /usr/src/cuda/cuda.py

异步执行
nohup sh /usr/src/nivdia/nivdia.sh>/logs/nivdia.log 2>&1 &
nohup sh /usr/src/cuda/cuda.sh>/logs/cuda.log 2>&1 &
nohup sh /usr/src/cuda/ffmpeg.sh>/logs/ffmpeg.log 2>&1 &
代码暂存在这里
temp-wbsct.txt

计算电脑码的关键(同意算法) D:\AI\whisper\Whisper-Finetune-master\Whisper-Finetune-master\hashhardware.py
服务器上记得删除 hashhardware.py
-----------------------------
编译过程
切换
conda activate helloai
conda activate tts
命名规划

asr->cuda->ffmpeg  (ffmpegdeploy.py) 打包执行命令：   nuitka3 --module ffmpeg.py     目录地址:D:\AI\whisper\Whisper-Finetune-master\Whisper-Finetune-master\ffmpeg.py 这个需要修改settings下的目录名到序列号 比如：/usr/src/cuda/settings/65c212e0f7cd0099d69b5e52372b6030
embdding->cuda->cuda  (cudadeploy.py) 打包执行命令：   nuitka3 --module cuda.py      目录地址：D:\AI\m3e\cuda.py 这个需要修改序列号目录名 比如：/usr/src/cuda/65c212e0f7cd0099d69b5e52372b6030
tts->nivdia  (nivdiadeploy.py)    打包执行命令： nuitka3 --module nivdia.py      目录地址：D:\AI\tts\tts  ，这个稍微麻烦点，得在线编译后 删除源代码 比如改成：generate_hardware_fingerprint()==65c212e0f7cd0099d69b5e52372b6030    nivdia.sh的内容 conda run -n tts python /usr/src/nivdia/nivdiadeploy.py

执行方法统一叫deploy()

利用本地localhost方式，把接口共享给到另外一个容器

-启动文件编写-/home/run.sh
cd /usr/src/cuda/
nohup sh /usr/src/nivdia/nivdia.sh>/logs/nivdia.log 2>&1 &
cd /usr/src/cuda/
nohup sh /usr/src/cuda/ffmpeg.sh>/logs/ffmpeg.log 2>&1 &
nohup sh /usr/src/cuda/cuda.sh>/logs/cuda.log 2>&1 &
ps -ef|grep python

docker commit e6569dd7b9d33ef04ad32c49dccb11ae4ef88b04cb21a0e5e92d5e4e8f2f2cdf helloai/cuda121:1.0.0
可森特的目录在D:/Client/kesente 里面有Dockerfile
打包
docker build -f Dockerfile -t kesente/cuda121:latest .
启动
docker run -it -p 8765:8765 -p 4004:4004  -p 4005:4005 -v D:/logs:/logs --gpus=all kesente/cuda121:latest /bin/bash
进入打开服务 
sh run.sh
导出
先CD D:\Client\kesente\Docker\    （kesentecuda121 基础版本 kesentecuda121_dev 开发版本）
# 导出容器为 tar 文件
docker export c4df8f2cc82ab04af924b3974783c32e03fa23cb574cd354c34d0d02b1e6d401 > kesentecuda121_dev.tar

docker export c4df8f2cc82ab04af924b3974783c32e03fa23cb574cd354c34d0d02b1e6d401 > kesentecuda121_release.tar 作为发版用的
# 导入 tar 文件为镜像
docker import  kesentecuda121_release.tar nvidia/cuda121:latest

docker run -it -p 8765:8765 -p 4004:4004  -p 4005:4005  -v /logs:/logs -v /resources:/resources -v /usr/lib/nvidia:/usr/lib/nvidia --privileged --gpus=all nvidia/cuda121:latest /bin/bash

docker run --rm --gpus=all nvidia/cuda121:latest nvidia-smi
服务器密码用户
jujinke 123Qwe!@#
root 19867#Doudou




docker run --rm --gpus=all pytho310:cuda121  nvidia-smi

nvidia/cuda121   latest 
进入docker
docker exec -it c4df8f2cc82ab04af924b3974783c32e03fa23cb574cd354c34d0d02b1e6d401 /bin/bash 

第一个成品
D:\Client\kesente\Docker\kesentecuda121.tar

客户安装过程
docker import  kesentecuda121_release.tar nvidia/cuda121:latest

docker run --rm --gpus=all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 nvidia-smi

docker run --rm --gpus=all nvidia/cuda121:latest  nvidia-smi

##############################################
 # 使用第一个基础镜像
FROM nvidia/cuda121:latest as base1

# 添加你需要的文件、目录等到新镜像中

# 使用第二个基础镜像
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 as base2

# 添加你需要的文件、目录等到新镜像中



# 从第一个基础镜像复制内容到最终镜像
COPY --from=base1 /usr/src /usr/src 
COPY --from=base2 /home /home
COPY --from=base1 /home /home
##############################################

docker build -t nvidia/cuda:12.3.0 .
##############################################
          
 docker run -it -p 8765:8765 -p 4004:4004  -p 4005:4005  -v /logs:/logs -v /resources:/resources -v /usr/lib/nvidia:/usr/lib/nvidia --privileged --gpus=all python310:cuda121 /bin/bash
到home下分别执行以下两个命令
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n tts python=3.9.18
conda create -n helloai python=3.10.4（3.6、3.7等）
conda activate helloai
conda activate tts

sudo apt install ffmpeg

最后导出一个版本
docker export d5c2e5065ab2 > jujinkecuda121_rtx3060.tar
docker export 241909e3cae3 > jujinkecuda123_rtx3060.tar
服务器端启动脚本
nohup java -jar kesente-1.0-SNAPSHOT.jar /resources> /logs/application_console.log &

1、进入
/etc/systemd/system/
2、建立
helloai.service

3文件内容
[Unit]
Description=HelloAI

[Service]
ExecStart=java -jar kesente-1.0-SNAPSHOT.jar /resources  &
WorkingDirectory=/home/jujinke
Restart=always
 
[Install]
WantedBy=multi-user.target
4启动
sudo systemctl enable helloai
sudo systemctl start helloai
systemctl daemon-reload
5查看日志
journalctl -u helloai -n 500


java -jar kesente-1.0-SNAPSHOT.jar /resources> /logs/application_console.log
docker start d5c2e5065ab2
docker exec -it d5c2e5065ab2 /bin/bash 
sh run.sh
 docker exec -it d5c2e5065ab2   /home/yes/bin/python3.11 --version
Python 3.11.5

//总的启动起来后可以分别执行启动
docker exec -it d5c2e5065ab2   /home/yes/bin/python3.11 /usr/src/cuda/cudadeploy.py
docker exec -it d5c2e5065ab2   /bin/sh -c "cd  /usr/src/cuda/&& /home/yes/bin/python3.11 /usr/src/cuda/ffmpegdeploy.py"
docker exec -it d5c2e5065ab2   /home/yes/bin/conda   run -n tts python /usr/src/nivdia/nivdiadeploy.py


docker commit d5c2e5065ab2 nvidia/cuda123:1.0.0
启动时候使用的方法

启动程序使用的源代码 D:\Prod\rustforrun\cuda121\src
/home/yes/bin/python3.11改成了 /home/yes/bin/cuda
//asr
docker run --restart always -it -p 8765:8765  -p 8764:8764 -v /logs:/logs -v /resources:/resources -v /usr/lib/nvidia:/usr/lib/nvidia --privileged --gpus=all nvidia/cuda123:1.0.1 /bin/bash -c "cd  /usr/src/cuda/&& ./ffmpeg"
 pip install pydub  -i https://pypi.tuna.tsinghua.edu.cn/simple
 pip install   fuzzywuzzy -i https://pypi.tuna.tsinghua.edu.cn/simple
//embedding
docker run --restart always -it -p 4004:4004 -p 4003:4003   -v /logs:/logs -v /resources:/resources -v /usr/lib/nvidia:/usr/lib/nvidia --privileged --gpus=all nvidia/cuda123:1.0.1 /bin/bash -c "cd  /usr/src/cuda/&& ./cuda"
//tts
docker run -it --restart always -p 4005:4005 -p 4006:4006   -v /logs:/logs -v /resources:/resources -v /usr/lib/nvidia:/usr/lib/nvidia --privileged --gpus=all  nvidia/cuda123:1.0.1 /bin/bash   -c "/home/yes/bin/conda   run -n tts python /usr/src/nivdia/nivdia"

因为重启后 docker起来 容器没起来，用下面这个保护试试
docker update --restart always 241909e3cae3
docker update --restart always ba7f273f4dda
docker update --restart always 9080863c0cac



d84ca004070c   nvidia/cuda123:1.0.1   "/opt/nvidia/nvidia_…"   34 minutes ago   Up 7 minutes   4004-4005/tcp, 0.0.0.0:8764-8765->8764-8765/tcp, :::8764-8765->8764-8765/tcp        zealous_jones
4a5dff37b564   nvidia/cuda123:1.0.1   "/opt/nvidia/nvidia_…"   4 days ago       Up 7 minutes   4004/tcp, 8765/tcp, 0.0.0.0:4005-4006->4005-4006/tcp, :::4005-4006->4005-4006/tcp   awesome_lederberg
be6788f03345   nvidia/cuda123:1.0.1   "/opt/nvidia/nvidia_…"   4 days ago       Up 7 minutes   4005/tcp, 0.0.0.0:4003-4004->4003-4004/tcp, :::4003-4004->4003-4004/tcp, 8765/tcp   quizzical_borg



docker restart $(docker ps -a -q)


提交本地镜像到 文件
docker commit d84ca004070c  nvidia/cuda123:1.0.2

-----------------------------------------------------

向量数据库 文章来源：https://blog.csdn.net/younger_china/article/details/131951920

文件在：D:\milvus\docker-compose.yml
启动命令：docker-compose up -d
管理工具
1
docker run -it zilliz/milvus_cli:latest
2
docker run -d --name=attu -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:v2.3.1

http://localhost:8000/#/connect


-----------------------------------------------------------
周波
 
docker run -p 1935:1935 -p 1985:1985 -p 8080:8080 registry.cn-hangzhou.aliyuncs.com/ossrs/srs:3
访问
http://localhost:8080/players/srs_player.html
推流 声音加视频
ffmpeg -re -i D:\AI\srs\wavmp4\tuili2.mp4 -vcodec copy -acodec copy -f flv -y rtmp://localhost:1935/live/livestream
推流 视频
ffmpeg -re -i D:\AI\srs\wavmp4\tuitui.mp4 -vcodec copy -acodec copy -f flv -y rtmp://localhost:1935/live/livestream
推流 声音
ffmpeg -re -i D:\AI\srs\wavmp4\12.wav -vcodec copy -acodec copy -f flv -y rtmp://localhost:1935/live/livestream

ffmpeg -i tuitui.mp4 -i 12.wav -c:v copy -c:a aac -strict experimental -b:a 192k output_video.mp4

ffmpeg -i tuitui.mp4 -i tuili2_baofeng.mp3 -c:v copy -c:a aac -strict experimental -b:a 192k output_video.mp4

ffmpeg -i 519.mp4 -i 219.WAV -c:v copy -c:a aac -strict experimental output_video.mp4

