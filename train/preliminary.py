from collections import defaultdict

from torch.utils.data import DataLoader

from utils.data_preprocess import split_dataset
from utils.dataset import mycollate, PreliminaryDataset


def train_model_preliminary(model, optimizer, tokenized_dataset, args):
    """
    :param model:
    :param optimizer:
    :param criterion:
    :param dataloader: 这个dataloader比较特殊，只是dict{op: }
    :param args:
    :return: 返回测试loss，或者考虑考虑别的
    """
    # 预留0.2的样本不参与元学习部分。这个参数比较随意，我就不放在args里了
    test_split_num = int(len(tokenized_dataset) * 0.8)
    sub_dataset, sub_test_dataset = tokenized_dataset[:test_split_num], tokenized_dataset[test_split_num:]
    dev_loss_for_example = defaultdict(list)  # 记录每个样例的测试loss
    example_to_data = {data["expression"]: data for data in sub_dataset} # 因为dev_loss_for_example是按照样例来记录的

    sub_dataset.shuffle()
    def get_sub_dataloader():
        train_dataset, _, dev_dataset = split_dataset(sub_dataset, args.split) # test相当于此刻的dev

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=mycollate)
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=mycollate)

        return train_dataloader, dev_dataloader

    # 训练的时候不希望更改模型的初始参数
    def inner_train(dataloader1, dataloader2) -> float:
        whole_loss = 0

        with higher.innerloop_ctx(
                model,
                optimizer,
                copy_initial_weights=True,
                track_higher_grads=False,
        ) as (fmodel, diffopt):
            for epoch in range(args.epoch):
                for batch in dataloader1:
                    batch = {k: v[0].to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    diffopt.step(fmodel(**batch)["loss"])

            with torch.no_grad():
                for i, batch in enumerate(dataloader2):
                    # print(batch)
                    batch = {k: v[0].to(args.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    outputs = fmodel(**batch)
                    whole_loss += outputs["loss"].detach().cpu()

        return whole_loss

    # 这里开始执行训练
    for _ in range(args.iterations_per_subset):
        train_dataloader, dev_dataloader = get_sub_dataloader()
        whole_loss = inner_train(train_dataloader, dev_dataloader)
        for i, batch in enumerate(dev_dataloader):
            # 遍历每个样例，记录测试loss
            for example in batch["expression"]:
                dev_loss_for_example[example].append(whole_loss)

    # 比较用全sub训练和sub的前80%训练的效果，在test上测试
    #1. 使用sub_dataset训练，sub_test_dataset测试
    sub_dataloader = DataLoader(sub_dataset, batch_size=args.batch_size, collate_fn=mycollate)
    sub_test_dataloader = DataLoader(sub_test_dataset, batch_size=args.batch_size, collate_fn=mycollate)

    loss_all_dataset = inner_train(sub_dataloader, sub_test_dataloader)

    #2. sub前80%
    for example in dev_loss_for_example:
        dev_loss_for_example[example] = sum(dev_loss_for_example[example]) / len(dev_loss_for_example[example])

    # 排序选loss最小的80%
    sorted_dev_loss = sorted(dev_loss_for_example.items(), key=lambda x: x[1])
    select_top_percentage = int(len(sorted_dev_loss) * args.select_top_percentage)
    selected_examples = [example_to_data[example] for example, _ in sorted_dev_loss[:select_top_percentage]]
    selected_dataset = PreliminaryDataset(selected_examples)

    selected_dataloader = DataLoader(selected_dataset, batch_size=args.batch_size, collate_fn=mycollate)

    loss_select = inner_train(selected_dataloader, sub_test_dataloader)

    print("全数据微调的效果是：", loss_all_dataset)
    print("部分数据微调的效果：", loss_select)
    print("选择比例", args.select_top_percentage)
    print("=====================")