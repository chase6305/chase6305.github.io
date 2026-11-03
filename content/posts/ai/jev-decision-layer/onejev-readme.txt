OneJev：概率复算、请求与响应解读
=============================

客户端使用 Python 3.10+ 标准库。解压后进入 onejev-examples/ 目录。

1. 先复算构造的数值，不需要模型或网络：

   python3 -B onejev_examples.py math

   三选一概率约为 0.8、0.1、0.1；Choice confidence 为 0.7；
   交叉熵加 Brier 的损失约为 0.2831435513。
   onejev-math-results.json 保存相同构造输入的计算结果。

2. 查看文字请求，不发送 HTTP：

   python3 -B onejev_examples.py request

   任务是启用桌面通知，文字观测描述开关目前关闭。

3. 不启动服务，也可以读取构造的响应节选：

   python3 -B onejev_examples.py read-response onejev-response-example.json

   该 JSON 的概率是构造值，经固定版本的答案组装代码生成，不是模型推理结果。
   它省略 usage，并把模型名标为 constructed-example-not-model-inference。
   四选一分布为 0.8、0.1、0.05、0.05，Choice confidence 为 0.733333；
   三级进度分布为 0.05、0.9、0.05，Score 为 1.0，Score confidence 为 0.85。
   read-response 只读取本例 done、next、progress 三题的字段，不发送 HTTP。

4. 启动模型服务后，发送文字与截图请求：

   python3 -B onejev_examples.py ask
   python3 -B onejev_examples.py ask --image onejev-settings-screen.png

   默认服务地址为 http://127.0.0.1:8000；可以用 --url 修改。
   默认服务名为 OneJev1.6-0.8B；可以用 --model 指定 /v1/models 中的名字。
   默认 HTTP 超时为 120 秒；可以用 --timeout 修改。

   截图是构造的设置页，通知开关已经开启。截图请求保留任务和媒体引用，
   不把文字例子中的开关状态一同发给模型。两个例子都有完成 Noul、
   下一步 Choice 和三级进度 Score；进度下标为 0、1、2，不是百分比。
   ask 打印服务器返回的全部 JSON 字段，不执行其中选中的界面动作。

   保存真实响应后，用同一命令读取：

   python3 -B onejev_examples.py ask > response.json
   python3 -B onejev_examples.py read-response response.json

   字段读取不是完整的执行门控；候选白名单、整个分布、观测时效和执行前条件
   仍由业务程序校验。read-response 针对本例的三个问题和 0–2 进度范围，
   修改问题 ID 或等级数后，应相应修改读取逻辑。

推理服务与客户端是两部分。NVIDIA GPU 路线可在独立环境中安装
Flide 0.1.7 和匹配的 torchvision，再启动：

   flide serve --model OmniJev/OneJev1.6-0.8B --host 127.0.0.1 --port 8000

文章另述 GGUF / llama.cpp 路线。模型、引擎、量化与权重 revision
都影响复现条件；上述数值复算不是模型准确率或校准效果测试。

文章及完整安装步骤：
https://chase6305.github.io/posts/ai/jev-decision-layer/#onejev

OneJev 仓库版本：d18d82243d82062af9c6c1bed999eaec5f39cbe0
0.8B 权重版本：c89933c277629b707895bff752c61d2b0664c85e
