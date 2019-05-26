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
> * 实现网络结构：2*CNN+1*256fc_net+output
> * 处理单帧图像输入；
> * 贪心率等比数列下降；
> * 经验回放策略

**- 期望的新功能：**
> * 连续4帧叠加输入（**有难度**）
> * 贪心率改为线性下降

