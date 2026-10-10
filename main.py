import torch
import torch.optim as optim
from Encoder import Encoder


def run_sanity_check():
    print("====== 🚀 開始執行 Encoder 健壯性檢查 ======")

    # 1. 設定測試超參數 (通常 Hidden Size 會等於 Embed Size)
    vocab_size = 200
    embed_size = 512
    hidden_size = 512
    max_length = 20
    head = 8

    # 2. 測試模型生成
    try:
        model = Encoder(vocab_size, embed_size, max_length, hidden_size, head)
        print("✅ 成功實例化 Encoder 模型。")
    except Exception as e:
        print(f"❌ 模型建立失敗: {e}")
        return

    # 3. 準備虛擬輸入數據 (Batch Size = 4, 序列長度 = 20)
    batch_size = 4
    seq_len = 20

    # 在 0 到 vocab_size 之間隨機生成整數 Token ID
    dummy_input = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    print(f"ℹ️ 輸入張量維度 (Shape): {dummy_input.shape}")

    # 4. 測試前向傳播 (Forward Pass)
    try:
        output = model(dummy_input)
        print("✅ 前向傳播成功完成。")
        print(f"ℹ️ 輸出張量維度 (Shape): {output.shape}")

        # 驗證輸出維度是否為 [Batch, Seq_Len, Embed_Size]
        expected_shape = (batch_size, seq_len, embed_size)
        assert output.shape == expected_shape, f"❌ 維度不符！預期為 {expected_shape}，但得到 {output.shape}"
        print("✅ 輸出維度完全符合預期。")

    except Exception as e:
        print(f"❌ 前向傳播失敗: {e}")
        return

    # 5. 測試反向傳播 (Backward Pass)
    try:
        # 計算一個簡單的純量 Loss 來觸發梯度計算
        loss = output.mean()
        loss.backward()
        print("✅ 反向傳播與梯度計算成功完成。")
    except Exception as e:
        print(f"❌ 反向傳播失敗 (可能存在非連通圖或張量斷裂): {e}")
        return

    print("====== 🎉 恭喜！所有 Sanity Check 順利通過！ ======")


if __name__ == "__main__":
    run_sanity_check()
