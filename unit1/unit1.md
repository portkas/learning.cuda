---
title: 基于cuda的异构并行计算
categories:
    - cuda
---
# 并行计算

相关概念：

* 数据相关性：一个指令处理前一个指令产生的数据。
* 并行性：任务并行，数据并行
* 带宽：单位时间内，可以处理的数据量（MB/s，GB/s）
* 吞吐量：单位时间内成功处理的运算数量（gflops）
* 计算机架构：单指令单数据(SISD)，单指令多数据（SIMD），多指令单数据（MISD），多指令多数据（MIMD）

数据并行程序设计：

* 第一步：把数据依据线程进行划分，使每个线程处理一部分数据。
  * 块划分：一组连续的数据被分到一个块内，每个数据块以任意次序被安排给一个线程，线程通常在同一时间只处理一个数据块。
  * 周期划分：更少的数据被分到一个块内，相邻的线程处理相邻的数据块，每个线程可以处理多个数据块。

计算机架构：
