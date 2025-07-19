import ray
from ray import tune
from ray.tune import CLIReporter

from transformers import AutoModelForSeq2SeqLM

from test_model import test_model
from .self_train import train_model_self_train


def train_tune(config, args, tokenizer, dataset, optimizer, model):
    """
    Ray Tune 的试验函数：根据 config 中的超参数训练模型、评估，并报告指标给 Tune。
    """
    # 更新超参数
    args.batch_size = config["batch_size"]
    args.learn_rate = config["learn_rate"]
    args.max_length = config["max_length"]
    args.epoch = config["epoch"]

    # 训练模型
    model = train_model_self_train(model, tokenizer, optimizer, dataset, args)

    # 在训练后评估模型
    accuracy, avg_loss = test_model(model, tokenizer, dataset, args, train_machine='ray')

    # 报告指标给 Ray Tune（Ray Tune 会根据这些指标进行调度和选择最佳试验）
    tune.report({"accuracy": accuracy, "avg_loss": avg_loss})



def tune_hyperparameters_ray(tokenizer, dataset, args, model_save_path, optimizer, model):
    """
    利用 Ray Tune 进行超参数网格搜索，每个试验分配一个 GPU。
    """
    # 初始化 Ray（若 Ray 已经初始化，可忽略 ignore_reinit_error 参数）
    ray.init(ignore_reinit_error=True)

    # 定义超参数搜索空间
    config = {
        "batch_size": tune.grid_search([args.batch_size]),
        "learn_rate": tune.grid_search([args.lr]),
        "max_length": tune.grid_search([args.max_length]),
        "epoch": tune.grid_search([args.epoch])
    }

    # 设置一个 CLI 报告器，可以在命令行中看到进度
    reporter = CLIReporter(
        metric_columns=["accuracy", "avg_loss", "training_iteration"]
    )

    # 调用 tune.run 开始超参数搜索
    analysis = tune.run(
        tune.with_parameters(train_tune, args=args, tokenizer=tokenizer, dataset=dataset, optimizer=optimizer, model=model),
        resources_per_trial={"gpu": 1},  # 每个试验分配 1 个 GPU；如果你的机器有多 GPU，就能实现不同试验分别在不同卡上运行
        config=config,
        metric="accuracy",
        mode="max",
        progress_reporter=reporter,
        storage_path=model_save_path,  # 日志和检查点保存目录
        name="tune_experiment"
    )

    # 输出最佳超参数组合
    best_config = analysis.get_best_config(metric="accuracy", mode="max")

    with open("best_config", "w", encoding="utf-8") as f:
        f.write(str(best_config))
    print("Best config: ", best_config)
