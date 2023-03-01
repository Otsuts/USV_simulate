# USV_simulate

这个项目是prp无人编队控制的项目，主要基于UE5实现了单船和多船的无人艇控制

其中TD3和SAC是两种基于连续动作空间的算法，DQN把连续的动作空间拆散，使得训练更加容易，Team则基于规则实现了一个简单的编队控制

需要配合UE5使用，在UE5中用TCP通信来实现python进程和UE进程的通信

UE环境的搭建可以参考：

https://www.bilibili.com/video/BV19Y4y1e7KQ/?spm_id_from=333.337.search-card.all.click

但是这个项目没有给出UE源文件

结项之后把我们的船给链接上来
