import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm


class Qk_hat:
    def __init__(self):
        pass

    def make_Qk_hat_df_with_known_Qk(
        self,
        demand_df: pd.DataFrame,
        Qks: np.ndarray,
        service_level: float,
    ) -> pd.DataFrame:
        """
        計算每期的補貨量預估 (Qk_hat)。

        參數:
        ----------
        demand_df : pd.DataFrame
            每列為時間點 (t=0..T-1)，每欄為樣本 (N)。shape=(T, N)
        Qks : np.ndarray
            已知的模擬補貨量分布樣本，shape=(T-1, N, simulation_times)
        service_level : float
            服務水準 (例如 0.95 表示 95%)。

        回傳:
        ----------
        pd.DataFrame
            每列為樣本 (sample_0..sample_{N-1})，每欄為 Qk_hat_{2}..Qk_hat_{T}。

        範例 (Example):（只取一個 item 舉例）
        ----------
        假設:
            T = 6  # t = 0 ~ 5（共6期），我們會建立 Qk_hat_2 ~ Qk_hat_6（共5期）
            N = 1  # 1 個 item
            simulation_times = 3

            demand_df (shape 1×6):
                0   1   2   3   4   5
            0  20  13  13  12  18   8

            代表 t=0 的需求是 20
            t=1 的需求是 13
            ...

            Qks (shape 5×1×3):  // 這是對應 t=2~6 的模擬分布樣本
            [[[28.724, 16.194, 21.595]],
            [[18.753, 27.311,  9.699]],
            [[18.388, 18.080, 25.669]],
            [[14.501, 19.138, 15.611]],
            [[20.211, 22.914, 14.497]]]

        計算過程:
            - t=2 (計算 Qk_hat_2):
            x_observed = [20] → sum=20  // 只取 t=1
            Qk_percentile = 第95百分位 ≈ 28.0112
            Qk_hat_2 = 20 + 28.0112 = 48.0112

            - t=3 (計算 Qk_hat_3):
            x_observed = [20,13] → sum=33 // 取 t=1,2
            Qk_percentile ≈ 26.4548
            Qk_hat_3 = 33 + 26.4548 = 59.4548

            - t=4 (計算 Qk_hat_4):
            x_observed = [20,13,13] → sum=46 // 取 t=1,2,3
            Qk_percentile ≈ 24.9408
            Qk_hat_4 = 46 + 24.9408 = 70.9408

            - t=5 (計算 Qk_hat_5):
            x_observed = [20,13,13,12] → sum=58 // 取 t=1,2,3,4
            Qk_percentile ≈ 18.7851
            Qk_hat_5 = 58 + 18.7851 = 76.7851

            - t=6 (計算 Qk_hat_6):
            x_observed = [20,13,13,12,18] → sum=76 // 取 t=1,2,3,4,5
            Qk_percentile ≈ 22.6438
            Qk_hat_6 = 76 + 22.6438 = 98.6438

        最終輸出:
            Qk_hat DataFrame (1×5):
                Qk_hat_2   Qk_hat_3   Qk_hat_4   Qk_hat_5   Qk_hat_6
            sample_0  48.0112     59.4548     70.9408     76.7851     98.6438
            Qk_hat DataFrame (1×4):
                Qk_hat_2   Qk_hat_3   Qk_hat_4   Qk_hat_5
            sample_0  48.0112     59.4548     70.9408     76.7851
        """

        N, T = demand_df.shape
        Qk_hat_matrix = np.zeros((N, T - 1))

        for t_index in range(T - 1): 
            Qks_t = Qks[t_index, :, :]  # shape: (3, simulation_times)

            for item_idx in range(N):

                # 取出單一 item 的所有 Qk，並且計算 Qk 的 service_level 分位數
                qk_dist = Qks_t[item_idx]  # shape: (simulation_times,)
                Qk_percentile = np.percentile(qk_dist, service_level * 100)

                # 現在是計算 t=k 的 Qk hat, 因此取出已經觀測到的值，是 1 ~ k-1 的真實值
                x_observed = demand_df.iloc[item_idx, : t_index + 1].values
                Qk_hat_matrix[item_idx, t_index] = x_observed.sum() + Qk_percentile

        time_columns = [f"Qk_hat_{t+2}" for t in range(T - 1)]
        return pd.DataFrame(Qk_hat_matrix, columns=time_columns)

