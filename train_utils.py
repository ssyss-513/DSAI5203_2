import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def get_dataloaders(dataset, test_set, batch_size, num_workers=4, pin_memory=True):
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        drop_last=False
    )
    test_loader = DataLoader(
        test_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    return train_loader, test_loader

def _prepare_frame(frame, device, frame_layout):
    """Convert frame layout for model forward.

    frame layouts:
    - 'btc': (B, T, C), no transpose
    - 'bct': (B, C, T), no transpose
    - 'tbc': (T, B, C), generated from (B, T, C) by transpose(0, 1)
    """
    if frame_layout == "tbc":
        return frame.transpose(0, 1).to(device)
    if frame_layout in ("btc", "bct"):
        return frame.to(device)
    raise ValueError(f"Unsupported frame_layout: {frame_layout}")

def plot_training_history(train_losses, train_accs, test_accs, model_name="Model"):
    clear_output(wait=True)
    plt.figure(figsize=(12, 6))

    if model_name:
        plt.suptitle(model_name, fontsize=16)

    # 绘制 Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='#e74c3c', lw=2)
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc', color='#3498db', lw=2)
    plt.plot(test_accs, label='Test Acc', color='#2ecc71', lw=2)
    plt.title('Accuracy Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def train_and_eval_visualized(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    save_dir,
    device,
    model_name="Model",
    frame_layout="tbc",
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据记录本
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }

    start_epoch = 0
    best_acc = 0
    latest_path = os.path.join(save_dir, 'latest_model.pth')

    # 加载存档与历史记录
    if os.path.exists(latest_path):
        print(f"🔄 检测到存档，正在恢复...")
        checkpoint = torch.load(latest_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        # 恢复历史曲线数据
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"✅ 成功恢复！从第 {start_epoch + 1} 轮继续，当前最佳: {best_acc:.4f}")

    print("🚀 开始训练...")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for frame, label in train_loader:
            frame = _prepare_frame(frame, device, frame_layout)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(frame)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # 统计
            batch_size = label.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += (output.argmax(1) == label).sum().item()
            total_samples += batch_size

        # 计算本轮指标
        epoch_loss = running_loss / total_samples
        epoch_train_acc = running_corrects / total_samples

        # 验证环节
        model.eval()
        test_corrects = 0
        total_test = 0
        with torch.no_grad():
            for frame, label in test_loader:
                frame = _prepare_frame(frame, device, frame_layout)
                label = label.to(device)
                output = model(frame)
                test_corrects += (output.argmax(1) == label).sum().item()
                total_test += label.size(0)

        epoch_test_acc = test_corrects / total_test

        # 更新记录
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_acc'].append(epoch_test_acc)

        # --- 实时绘图 ---
        plot_training_history(history['train_loss'], history['train_acc'], history['test_acc'], model_name=model_name)

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Time: {epoch_time:.1f}s | "
              f"Loss: {epoch_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_test_acc:.4f}")

        # --- 保存逻辑 ---
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(epoch_test_acc, best_acc),
            'history': history  # 将历史记录存入模型文件
        }

        torch.save(state, latest_path)
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(state, os.path.join(save_dir, 'best_model.pth'))
            print(f"⭐ 最佳模型已更新 (Acc: {best_acc:.4f})")

    print(f"🎉 训练结束！最高准确率: {best_acc:.4f}")
