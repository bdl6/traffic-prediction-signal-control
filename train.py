import torch
import torch.nn as nn
import torch.optim as optim
from models.data_process import load_data, normalize_data, build_spatiotemporal_dataset
from models.data_process import split_dataset, to_tensor
from models.graph_wavenet import GraphWaveNet

def main():
    # 超参数
    batch_size = 32
    epoch_num = 80
    lr = 0.001
    node_num = 12
    his_steps = 24
    pre_steps = 12

    # 1.加载与处理数据
    data = load_data("./data/traffic_data.csv")
    data_norm, scaler = normalize_data(data)
    X, Y = build_spatiotemporal_dataset(data_norm, his_steps, pre_steps, node_num)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = split_dataset(X, Y)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = to_tensor(X_train, X_val, X_test, Y_train, Y_val, Y_test)

    # 2.初始化模型、损失、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphWaveNet(num_nodes=node_num, in_dim=1, hid_dim=32, embed_dim=10, pre_steps=pre_steps).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3.开始训练
    print("===== 开始训练 Graph WaveNet =====")
    model.train()
    for epoch in range(epoch_num):
        total_loss = 0.0
        idx = 0
        while idx + batch_size < len(X_train):
            batch_x = X_train[idx:idx+batch_size].to(device)
            batch_y = Y_train[idx:idx+batch_size].to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            idx += batch_size

        avg_loss = total_loss / (len(X_train) // batch_size)
        print(f"Epoch [{epoch+1}/{epoch_num}] | Train Loss: {avg_loss:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "wavenet_best.pth")
    print("训练完成，模型已保存为 wavenet_best.pth")

if __name__ == "__main__":
    main()