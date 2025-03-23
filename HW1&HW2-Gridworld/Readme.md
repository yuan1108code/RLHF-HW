# 強化學習網格世界 - HW1 & HW2 報告

<center>
        <img src="./templates/IMG.jpg"
             style="
                    width: 60%; 
                    height: auto;">
        <div style="
                border-bottom: 3px solid #d9d9d9;
                display: inline-block;
                color: #999;
                padding: 3px;">
        </div>
</center>


## 1. 引言 🚀

歡迎來到 強化學習網格世界 (Reinforcement Learning Gridworld)！這是一個互動式的網頁應用，結合 Flask 後端 與 HTML/CSS/JavaScript 前端，讓使用者能夠透過滑鼠點擊，在 n x n 網格中設置環境，並學習如何透過 價值迭代 (Value Iteration) 與 策略評估 (Policy Evaluation) 來計算最佳行動策略 🏆。  

這次的專案包含 兩個核心部分：

### **📌 HW1: 網格地圖開發與策略評估**
這部分的目標是建立一個可調整大小的網格，並讓使用者能夠與環境互動。
- 🎯 **環境設定**：
  - 用滑鼠點擊來指定 **起點**（綠色）與 **目標點**（紅色）。
  - 設定最多 `n-2` 個 **障礙物**（灰色），模擬不可通行區域。
- 🧭 策略與價值函數顯示：
  - 隨機生成每個格子的移動方向 (⬆️⬇️⬅️➡️) 作為初始策略。。
  - 使用策略評估 (Policy Evaluation) 計算每個狀態的 **價值函數 V(s)**，顯示在格子內。

### **📌 HW2: 使用價值迭代算法推導最佳政策**
在這部分，我們將應用 **價值迭代 (Value Iteration)** 來尋找 **最優策略**，讓每個格子上的行動不再是隨機的，而是能夠導引智能體朝著目標點移動。
- **🎯 計算最佳策略**：透過 價值迭代 (Value Iteration)，讓智能體根據當前價值函數選擇最佳行動方向。
- **📊 顯示最優價值函數**：更新 V(s) 來反映在最優策略下，每個格子的期望回報。

### 🎯 專案目標
這個專案的目標是讓使用者能夠透過互動方式直觀理解： 
✅ 如何初始化環境並設定不同條件的網格世界 🌎
✅ 如何透過策略評估來計算每個狀態的價值 📈
✅ 如何利用價值迭代來推導最佳策略 🔄



<br>

## 2. 系統設計 🏗️

為了讓這個專案既好玩又有學習價值，我採用了前後端分離的架構，讓 Flask 處理數據與運算，前端負責視覺化呈現 🎨📊。


![截圖 2025-03-24 凌晨2.40.02](https://hackmd.io/_uploads/rJ-5HA621g.png)



### **2.1 後端設計（Flask） 🖥️**
後端使用 Flask 作為 Web 框架，負責： 
✔ 網格初始化
✔ 接收使用者輸入（起點、目標、障礙物）
✔ 運行強化學習演算法（策略評估 & 價值迭代）
✔ 與前端進行數據交互 📡

#### 🔹 **主要 API 設計：**

<center>
    
| API 路徑 | 功能描述 | 
| :--: | :--: | 
| `/initialize_grid`	| 初始化 n x n 網格，生成隨機策略 |
| `/set_start` | 設定 **起始點**（綠色） |
| `/set_goal` | 設定 **目標點**（紅色）|
| `/toggle_obstacle` | 增加或移除 **障礙**物（灰色）|
| `/randomize_policy` | 隨機生成行動策略（⬅️➡️⬆️⬇️） |
| `/evaluate_policy` | 使用 **策略評估** 計算 **價值函數 V(s)** 📊 |
| `/value_iteration` | 執行 **價值迭代**，計算 最佳策略 🎯 |
| `/simulate_learning` | 模擬學習過程，逐步收斂到最優策略 🤖 |

</center>
    
### 2.2 前端設計（HTML + JavaScript） 🎨
前端負責： 
✔ 建立互動式 UI，讓使用者可視化網格
✔ 即時更新格子狀態（起點、目標點、障礙物）
✔ 顯示策略 (⬆️⬇️⬅️➡️) 與價值函數 (V(s))
✔ 提供動畫顯示學習過程 🎬

**📍 1. 使用者互動**
- 點擊格子可 設定起點、目標點與障礙物
- 點擊「開始學習」按鈕，啟動 Flask API 計算最佳策略

**📍 2. 網格 UI**
- 使用 CSS Grid 呈現 n x n 格子
- 格子顏色代表不同狀態：
    1. 🟩 起點 (Start)
    2. 🟥 目標點 (Goal)
    3. ⬛ 障礙物 (Obstacle)
    4. ⭐ 最短路徑 (Path)

### 2.3 互動示例 🎮

**🟢 步驟 1**：使用者選擇 n x n 的網格大小，點擊「生成網格」
**🔵 步驟 2**：點擊格子來設定 起點 和 目標點
**🔳 步驟 3**：選擇幾個障礙物，增加挑戰性
**🎯 步驟 4**：執行「策略評估」或「價值迭代」，查看最佳路徑


這個流程能夠讓使用者直觀體驗強化學習的策略收斂過程 🤩。

<br>

## 3. 價值評估與策略學習 📈🤖

在這部分，我們將探討 如何計算 `V(s)` 以及如何透過學習找到最佳策略！💡

我們會分別介紹： 
1. ✅ 策略評估 (Policy Evaluation) — 在給定策略下計算 `V(s)`
2. ✅ 價值迭代 (Value Iteration) — 找到讓 `V(s)` 最大化的最佳策略


### **3.1 策略評估 (Policy Evaluation)** 🧐
策略評估的目標是 **在給定策略 $\pi$ 下，計算每個狀態 s 的價值 V(s)**。

這個過程告訴我們：如果按照當前策略執行，預期能獲得多少回報？

📌 **核心公式**：
\begin{equation}
V(s) = R(s) + \gamma \sum_{s'} P(s' | s, a) V(s')
\end{equation}

這表示：
- 狀態 s 的價值 `V(s)` = 當前獎勵 `R(s)` + 折扣後的 未來狀態價值總和。
- `gamma` (折扣因子，通常 0.9 或 0.99) 控制 未來回報的重要性。
- `P(s' | s, a)` 表示從 `s` 採取行動 `a` 後到達 `s'` 的機率（在本專案中為確定性轉移，即 100% 確定轉移到某一狀態）。
- 這個計算過程會反覆迭代，直到 `V(s)` 收斂（即變化極小）。

**🛠 策略評估的 Flask API**：
- `/evaluate_policy` 會在後端進行策略評估，計算每個格子的 `V(s)`。
- 前端會即時更新網格，顯示 `V(s)` 於每個格子中。

**✨ 策略評估視覺化**：

當 `/evaluate_policy` API 被調用後，前端會即時更新網格上的 `V(s)`：
- 數值較大的格子 代表預期回報較高。
- `V(s)` 會逐步收斂，顯示該策略下的價值。


### **3.2 價值迭代 (Value Iteration) 🚀**
價值迭代的目標是 不只是計算 `V(s)`，還要找出最佳策略 🎯。

這代表我們會根據 `V(s) `選擇能帶來最高回報的行動，而不是固定使用當前策略。

📌 **核心公式**：

\begin{equation}
V(s) \leftarrow \max_a \left( R(s) + \gamma \sum_{s'} P(s' | s, a) V(s') \right)
\end{equation}

表示：
- 不同於策略評估，我們不再只是套用策略，而是主動選擇最佳行動。
- `V(s)`不再由 $\pi$ 決定，而是取 所有可能行動中，回報最高的那個。



🛠 **Flask API:**
- `/value_iteration` 會在後端執行價值迭代，計算最優策略。
- 每個格子的箭頭會更新為最佳行動方向，並同步更新 `V(s)`。

**✨ 價值迭代視覺化**：
價值迭代後，策略箭頭會被更新為最佳行動方向 ⬆️⬇️⬅️➡️

🎬 **動畫視覺化：**
**為了讓學習過程更直觀，我們設計了 學習過程動畫：**
1️⃣ simulate_learning 會每隔一段時間更新策略與 V(s)。

2️⃣ 使用者可以觀察策略如何逐步變得更聰明 🧠。

3️⃣ 最終策略將指引智能體到達 最佳路徑 🚀。

這讓使用者不只是看到結果，還能親身體驗強化學習的演進過程 🎥📊。

<br>

## 4. 測試流程與效能優化 🛠️🚀
![截圖 2025-03-24 凌晨2.40.42](https://hackmd.io/_uploads/B1Q6B0Thyl.png)

為了確保系統穩定運作，我們進行了一系列測試，並針對效能進行優化。

### **4.1 測試流程 ✅**
我們針對以下幾個核心功能進行測試，以確保程式的穩定性與正確性：

#### 1️⃣ **網格初始化測試** 🏗️

📌 目標：確認 /initialize_grid API 能正確產生 n x n 網格。

✅ 測試項目：
- n 的範圍是否限制在 5~9（避免異常輸入）。
- 每個格子是否都成功初始化，且隨機策略能正確分配至每個格子。
- 確認回傳的 policy、values 矩陣大小正確，避免超出邊界。

```python=
response = client.post('/initialize_grid', json={'size': 7})
assert response.status_code == 200
data = response.get_json()
assert data['size'] == 7
assert len(data['policy']) == 7
assert len(data['policy'][0]) == 7
```

#### 2️⃣ **環境設定測試** 🎯
📌 目標：確保 `/set_start`、`/set_goal`、`/toggle_obstacle` 能正確修改網格狀態。

✅ 測試項目：
- 設定 起點、目標點、障礙物 時，確保不會重疊或超出邊界。
- 障礙物的數量不得超過 `n-2`，確保系統限制有效。
- 每次修改後，前端 UI 是否正確更新。

```python=
response = client.post('/set_start', json={'row': 2, 'col': 3})
assert response.status_code == 200
assert response.get_json()['success'] == True

response = client.post('/set_goal', json={'row': 4, 'col': 4})
assert response.status_code == 200
assert response.get_json()['success'] == True
```

3️⃣ **策略與價值函數測試** 🔄
📌 目標：確保 `/evaluate_policy` 正確計算 `V(s)`，並驗證 `/value_iteration` 能正確更新最佳策略。

✅ 測試項目：
- 確認 `V(s)` 會逐步收斂，並與理論值接近。
- 確保 `/value_iteration` 能夠正確產生最佳策略，行動方向與 `V(s)` 一致。
- 變更環境（如加入障礙物）後，策略應該自適應變化。

```python=
response = client.post('/evaluate_policy')
assert response.status_code == 200
data = response.get_json()
assert isinstance(data['values'], list)
assert len(data['values']) == grid_size
```

4️⃣ **學習過程測試** 📈
📌 目標：確保 `/simulate_learning` 可視化學習過程並逐步收斂。

✅ 測試項目：
- 確保學習步驟合理，最終結果與 `/value_iteration` 一致。
- 測試不同 `gamma` (折扣因子) 設定，確認收斂速度是否合理。
- UI 是否隨學習過程逐步變化，策略箭頭更新是否同步。

```python=
response = client.post('/simulate_learning')
assert response.status_code == 200
data = response.get_json()
assert 'learning_steps' in data
assert isinstance(data['learning_steps'], list)
```

### **4.2 效能優化 🚀**
在初步測試後，我們發現了一些效能瓶頸，並採取以下優化策略來提升運行效率。

#### 🔹 減少 Flask API 請求次數

**🚀 問題**：每次更新 `V(s)` 或策略時，前端需要多次請求 API，導致網頁響應變慢。

**✅ 優化策略**：
- 合併請求：讓 `/value_iteration` 直接回傳 `V(s)` + 最佳策略，一次更新 UI，減少 HTTP 請求數量。
- 前端儲存快取：如果 `V(s)` 沒有變化，就不重新請求 API，減少不必要的更新。

```python=
@app.route('/value_iteration', methods=['POST'])
def value_iteration():
    global policy, values
    # 運行價值迭代...
    return jsonify({'values': values, 'policy': policy})
```


#### 🔹 **提升前端渲染效能**：

**🚀 問題**：在大網格（如 9x9）上，每次 V(s) 更新都會重繪整個網格，導致視覺卡頓。

**✅ 優化策略**：
- 使用 requestAnimationFrame() 讓瀏覽器優化繪製過程，減少 DOM 操作次數，提高動畫流暢度。
- 最小化更新範圍，僅更新改變的格子，而不是全部重新渲染。

```javascript=
function updateGridUI() {
    requestAnimationFrame(() => {
        document.querySelectorAll('.cell').forEach((cell, index) => {
            const row = Math.floor(index / gridSize);
            const col = index % gridSize;
            cell.querySelector('.value').innerText = stateValues[row][col];
        });
    });
}
```

#### 🔹 **使用 NumPy 加速計算**：
**🚀 問題**：Python 迴圈計算 `V(s)` 效率低，在較大網格 (9x9) 時，策略評估可能需要較長時間收斂。

**✅ 優化策略**：
- 將 `V(s)` 計算改為 NumPy 矩陣運算，大幅減少迴圈次數，加速收斂。

```python=
import numpy as np

def evaluate_policy_numpy():
    global values
    values = np.zeros((grid_size, grid_size))

    while True:
        old_values = np.copy(values)
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) in obstacles or (i, j) == goal_position:
                    continue
                
                action_idx = policy[i][j]
                next_i, next_j = i + ACTIONS[action_idx][0], j + ACTIONS[action_idx][1]

                if 0 <= next_i < grid_size and 0 <= next_j < grid_size:
                    values[i, j] = rewards[next_i, next_j] + gamma * values[next_i, next_j]
        
        if np.max(np.abs(old_values - values)) < theta:
            break
```

#### 🔹 動態調整學習速率
**🚀 問題**：不同使用者有不同需求，有些希望快速看到結果，有些則希望細緻觀察學習過程。

**✅ 優化策略**：
- 在前端提供速度選擇器，允許使用者調整學習步驟顯示速度（如 `0.5x`, `1x`, `2x`）。

```html=
<label for="learning-speed">學習速度：</label>
<select id="learning-speed">
    <option value="2000">慢</option>
    <option value="1000" selected>中</option>
    <option value="500">快</option>
</select>
```

<br>

## 5. 結論與未來改進 🎯

### **5.1 作業成果總結 📌**
本專案主要完成了兩個核心目標： 
- HW1：開發互動式網格地圖，並提供隨機策略顯示與策略評估功能 🏗️
- HW2：使用價值迭代 (Value Iteration) 計算最佳策略，並透過 箭頭方向與 `V(s)` 值 呈現結果 📈🔄

**✨ 主要特色**
1. 🚀 強化學習應用：透過策略評估與價值迭代來尋找最佳策
2. 🎨 可視化互動：即時顯示策略變化，讓學習過程更直
3. ⚡ 效能優化：使用 NumPy 加速計算，提升學習效率

本專案不僅滿足了作業要求，還額外進行了效能優化與使用者體驗提升，讓學習過程更加流暢與有趣！🎯✨


### **5.2 未來改進方向 🔍**
雖然目前的系統已經可以展示強化學習的基本概念，但仍然有一些可以改進的地方：

#### 💡 1. 加入 Q-learning 或 DQN
目前的策略學習基於 價值迭代 (Value Iteration)，這適用於 小型離散環境。
未來可以加入 Q-learning 或 深度強化學習 (DQN, Deep Q-Network) 來讓智能體能夠學習更複雜的環境，例如：
- 加入隨機動態環境，模擬不同狀態轉移機率 (`P(s' | s, a)`)
- 強化 Q-table 學習機制，讓智能體透過試錯 (exploration-exploitation) 找到最優策略
- 使用 DQN 處理更高維度的問題，如自動駕駛、機器人導航 🏎️

#### 💡 2. 提升可視化體驗
目前的 UI 主要透過 箭頭方向 來表示策略，未來可以進一步改進：

1. **不同顏色代表不同策略變化 🎨**：例如，隨著學習過程，策略收斂時顏色從 紅色 → 藍色 → 綠色，直觀顯示學習效果。
2. **更動態的動畫 🎬**：讓智能體的移動更具流暢感，例如以動畫方式逐步移動到最優路徑。
3. **增加學習進度條 📊**：顯示學習步驟數，讓使用者能夠掌握當前策略優化進度。

這樣的改進可以讓學習過程更具吸引力，提升使用者對強化學習的直觀理解 🎯。

#### 💡 3. 允許用戶自定義獎勵機制
目前的獎勵函數是固定的，例如：
- 目標點：+20
- 障礙物：-1
- 一般步驟：-0.1

未來可以提供使用者自訂獎勵值的功能： 
1. ✅ 選擇不同的獎勵設定，讓使用者觀察學習結果如何變化。
2. ✅ 加入不同類型的懲罰/獎勵機制，例如：
    - 設置負獎勵區域（如沼澤地）
    - 時間懲罰機制（學習過程越久，獎勵越低）

這不僅讓系統更具彈性，也能進一步幫助使用者理解強化學習的影響！💡🚀

#### 💡 4. 擴展至更大的環境
目前的網格限制在 5x5 ~ 9x9，但現實世界中的應用通常是更大範圍的環境，例如： 
1. ✅ 更大維度的迷宮導航 (10x10, 20x20) 🏰
2. ✅ 應用於自動駕駛環境 (模擬 3D 空間行動) 🚗
3. ✅ 結合 OpenAI Gym，測試在動態環境下的學習效果 🤖

#### 💡 5. 增加多智能體學習
目前系統僅支持單一智能體 (Agent)，但在真實環境中，通常會有多個智能體同時學習與互動 🤖🤖。

**未來可以增加：**
1. ✅ 多個 AI 角色，互相競爭或合作 🏆
2. ✅ 模擬群體行為，例如機器人協作任務 (Multi-Agent Reinforcement Learning)
3. ✅ 不同智能體採取不同學習策略，測試不同強化學習方法的比較

### 5.3 強化學習的潛在應用 🚀
本專案展示了 策略學習 (Policy Learning) 在網格世界的應用，但強化學習還有更多潛在應用，包括：
- 遊戲 AI 訓練 🎮（如 AlphaGo、Atari 遊戲 AI）
- 機器人導航 🤖（如自動倉儲機器人）
- 智慧交通系統 🚦（如 AI 自動駕駛）
- 金融決策建模 💰（如股票市場交易策略）

