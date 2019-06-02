# -Gym-
使用gym框架进行算法验证和学习
## 第一版进度
- 实现RL类及其接口，可以直接进行q-learning学习，需传入actions，可修改参数
- 实现Q-table类及其接口，可以方便调用，需传入合法actions和可哈希的state
- 做了对象序列化，存储训练后的RL对象
- state定义（x_pos,time，done）
- reward定义 r
 
    v = x1 - x0
    x0 is the x position before the step
    x1 is the x position after the step

    c = c0 - c1
    c0 is the clock reading before the step
    c1 is the clock reading after the step

    d: a death penalty that penalizes the agent for dying in a state
    alive ⇔ d = 0
    dead ⇔ d = -15
    r = v + c + d
- action暂时用actions.py第一组定义

## 第1.1版进度
> * 加入了vx的判断，成为惩罚项。
> * state定义（x_pos,life，done）。
> * 长时间迭代可完成第一关


## DQN第1版进度
> * 实现网络结构：2CNN+1fc_net(256)+output(simple_movement)
> * 处理单帧图像输入；
> * 贪心率等比数列下降；
> * 经验回放策略

**- 期望的新功能：**
> * 连续4帧叠加输入（**有难度**）
> * 贪心率改为线性下降
> * **加入模型存储功能，一定iteration后存储**

## 对DQN的改进：
 - 网络改为TF组织
 - 连续4帧统一动作，连续4帧做网络输入
 - 需要改变网络参数、循环结构、短时记忆如何实现、或者直接改变预处理？
 - 80*80 or 84*84 ？
 - epoch/episode = 100; iteration = 10000
 - Double DQN：一个用来决定行为，一个用来估计价值
 > * update_target_net()
 > * save_weight()
 > * loss应该需要重写（参考github）
 - A3C（policy gradient相关）
 
 - 我想做的：
 > * DDQN
 > * 在输出中加入日志，例如epoch平均reward等等
 > * A3C随缘，看时间是否足够（据说可以加快速度），因为用到了policy gradient似乎效果更棒
 
 
## 对DQN的实际改进：
 - 图像预处理进行无损采样，降维为16*16
 - 加入大跳action，加速训练，防止由于大跳需要13步而需要非常长时间的迭代
 - 修改了一些超参数，看起来变得神奇了
 - 完善了模型的读写
 - 修改为ipynb格式，方便调试和用户交互，方便使用云计算平台
 - 加入了DDQN的设置，但是具体性能未知
 
**- 对QL进行了类似的优化：**
