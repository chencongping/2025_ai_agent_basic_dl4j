package com.ai.agent;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 这是一个使用 DeepLearning4J 训练 MNIST 手写数字识别模型的完整示例。
 * 它包含了模型配置、数据加载、模型训练、评估和基本的可视化。
 */
public class MnistClassifier {

    // 定义一个日志记录器，方便查看程序运行状态
    private static final Logger log = LoggerFactory.getLogger(MnistClassifier.class);

    public static void main(String[] args) throws Exception {

        // --- 1. 配置神经网络模型 ---
        log.info("开始配置神经网络...");
        int seed = 123; // 随机种子，确保结果可复现
        int batchSize = 64; // 每次训练的样本数
        int numEpochs = 5; // 训练的总轮数

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(0.001)) // 使用 Adam 优化器，学习率 0.001
                .list()
                // 第一层：全连接层 (Dense Layer)
                .layer(0, new DenseLayer.Builder()
                        .nIn(784) // 输入大小：28x28 像素 = 784
                        .nOut(256) // 输出 256 个神经元
                        .activation(Activation.RELU) // 激活函数：ReLU
                        .build())
                // 第二层：输出层 (Output Layer)
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(256)
                        .nOut(10) // 输出 10 个类别 (0-9)
                        .activation(Activation.SOFTMAX) // 激活函数：Softmax，得到概率分布
                        .build())
                .build();

        // 根据配置创建模型实例
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        log.info("神经网络配置完成，模型已初始化。");


        // --- 2. 准备数据 ---
        log.info("正在加载 MNIST 数据集...");
        String localDataSetPath = "D:\\5_GitHub\\datasets\\mnist\\";
        // 训练集迭代器

        DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, seed);
        // 测试集迭代器
        DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, seed);
        log.info("MNIST 数据集加载成功。");


        // --- 3. (可选但推荐) 设置可视化UI ---
        log.info("启动训练可视化 UI...");
        UIServer uiServer = UIServer.getInstance();
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        // 为模型添加监听器，将训练数据发送到UI
        model.setListeners(new StatsListener(statsStorage));
        log.info("UI 已启动。请在浏览器中打开: http://localhost:9000");


        // --- 4. 训练模型 ---
        log.info("开始训练模型，共 {} 轮...", numEpochs);
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- 正在进行第 {} 轮训练 ---", (i + 1));
            // fit() 方法是训练的核心，它会处理整个批次的数据
            model.fit(trainIter);
            log.info("第 {} 轮训练完成。", (i + 1));

            // 在每轮训练后，在测试集上评估模型性能
            log.info("--- 在测试集上评估模型 ---");
            Evaluation eval = model.evaluate(testIter);
            // 打印评估结果（包含准确率等信息）
            log.info(eval.stats());

            // 重要：重置迭代器，以便下一轮训练和评估可以重新使用数据
            trainIter.reset();
            testIter.reset();
        }
        log.info("所有训练轮次已完成！");


        // --- 5. (可选) 保存模型 ---
        // ModelSerializer.writeModel(model, "mnist-model.zip", true);
        // log.info("模型已保存到 mnist-model.zip");


        // --- 6. 关闭资源 ---
        uiServer.stop();
        log.info("程序执行完毕，UI服务已关闭。");
    }
}