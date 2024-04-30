### AlexNet 논문을 TensorFlow와 PyTorch로 구현한 후, Torch-TensorRT / PyTorch JIT Trace 모드 적용 전후의 inference 속도 차이를 비교
Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Communications of the ACM 60.6 (2017): 84-90.

AlexNet 모델 아키텍처는 다음과 같으며, 데이터셋은 Cifar10을 Optimizer는 Adam을 사용한다.
![image](https://github.com/sophkim/cnn_pytorch_ex/assets/86454825/e35ce073-b207-4e10-a41a-3f6c7766fde7)<br>
<Input layer - Conv1 - MaxPool1 - Norm1 - Conv2 - MaxPool2 - Norm2 - Conv3 - Conv4 - Conv5 - Maxpool3 - FC1- FC2 - Output layer>



#### 1. TensorFlow 코드
#### 2. Pytorch 코드
#### 3. Torch-TensorRT 적용 후 inference 비교
#### 4. PyTorch JIT Trace 적용 후 inference 비교
#### 5. PyTorch, TensorFlow 논문 Summary (PyTorch: An Imperative Style, High-Performance Deep Learning Library / TensorFlow: A System for Large-Scale Machine Learning)
