import os
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections importdeque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 모델 아키텍처 (안정성을 위해 풀링 방식 변경)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiAssetTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, nhead=8, num_layers=2):
        num_coins = observation_space.shape[0]
        features_dim = num_coins * d_model
        super().__init__(observation_space, features_dim=features_dim)
        self.d_model, self.num_coins = d_model, num_coins
        _, seq_len, feat_dim = observation_space.shape
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size * self.num_coins, x.shape[2], x.shape[3])
        projected = self.input_proj(x_reshaped)
        projected_with_pos = self.pos_encoder(projected)
        encoded = self.transformer(projected_with_pos)
        # ⭐ 안정성 향상을 위해 마지막 스텝 대신 '평균 풀링(Mean Pooling)'으로 변경
        aggregated_features = encoded.mean(dim=1) 
        output = aggregated_features.reshape(batch_size, self.num_coins * self.d_model)
        return output

class MultiAssetPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
            features_extractor_class=MultiAssetTransformerExtractor,
            features_extractor_kwargs=dict(d_model=128, nhead=8, num_layers=4))

# -----------------------------------------------------------------------------
# 2. 거래 환경 (현실성, 전략 정교화, 리스크 관리 모두 적용)
# -----------------------------------------------------------------------------
class AdvancedTradingEnv(gym.Env):
    def __init__(self, data_folder, window_size=64, max_coins=10, initial_balance=10000.0, 
                 leverage_tiers=[1, 2, 5], position_size_tiers=[0.1, 0.25, 0.5], 
                 trade_fee=0.001, slippage_rate=0.0005, stop_loss_pct=0.1):
        super().__init__()
        
        # --- 1. 데이터 로드 ---
        all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
        df_list = []
        for file in tqdm(all_files[:max_coins], desc=f"파일 로딩 중 ({max_coins}개 코인)"):
            coin_name = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['timestamp'])
            df['coin'] = coin_name
            df_list.append(df)
        full_df = pd.concat(df_list, ignore_index=True)
        pivot_df = full_df.pivot_table(index='timestamp', columns='coin', values=['open', 'high', 'low', 'close', 'volume'])
        pivot_df.ffill(inplace=True); pivot_df.bfill(inplace=True); pivot_df.dropna(inplace=True)
        
        self.coin_names = sorted(pivot_df.columns.get_level_values(1).unique().tolist())
        self.num_coins = len(self.coin_names)
        prices_raw = pivot_df.values.reshape(len(pivot_df), 5, self.num_coins)
        self.prices = np.transpose(prices_raw, (2, 0, 1))

        # --- 2. 환경 변수 설정 ---
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.leverage_tiers = leverage_tiers
        self.position_size_tiers = position_size_tiers # 가용 자산 대비 투자 비율
        self.trade_fee = trade_fee
        self.slippage_rate = slippage_rate
        self.stop_loss_pct = stop_loss_pct

        # --- 3. 액션 스페이스 재설계 ---
        # [코인, 행동, 레버리지, 투자비중] 선택
        # 행동: 0:유지, 1:롱, 2:숏, 3:포지션 종료
        self.action_space = gym.spaces.MultiDiscrete([
            self.num_coins, 4, len(self.leverage_tiers), len(self.position_size_tiers)])
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_coins, window_size, 5), dtype=np.float32)
        
        self.reset()

    def reset(self):
        self.index = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {name: {'type': 0, 'leverage': 0, 'entry_price': 0, 'size': 0} for name in self.coin_names}
        return self._get_obs()

    def _get_obs(self):
        return self.prices[:, self.index - self.window_size:self.index, :]

    def step(self, action):
        coin_idx, action_type, leverage_idx, size_idx = action
        
        # --- 1. 이전 스텝 가치 기록 ---
        prev_portfolio_value = self.portfolio_value

        # --- 2. 행동 수행 ---
        self._execute_trade(action)

        # --- 3. 다음 스텝으로 이동 및 현재 가치 계산 ---
        self.index += 1
        self.portfolio_value = self._calculate_portfolio_value()
        
        # --- 4. 리스크 관리(Stop-Loss) 확인 ---
        self._check_stop_loss()
        
        # --- 5. 보상 계산 (포트폴리오 가치 변화율) ---
        reward = (self.portfolio_value / prev_portfolio_value) - 1
        
        done = self.index >= self.prices.shape[1] - 2 # 인덱싱 오류 방지
        
        # 전체 자산 청산 페널티 (선택적)
        if self.portfolio_value < self.initial_balance * 0.2:
            done = True
            reward = -1 # 파산 시 큰 페널티

        return self._get_obs(), reward, done, {}

    def _execute_trade(self, action):
        coin_idx, action_type, leverage_idx, size_idx = action
        coin_to_act = self.coin_names[coin_idx]
        current_price = self.prices[coin_idx, self.index, 3] # 종가
        
        if action_type == 1 or action_type == 2: # 롱 또는 숏
            # 기존 포지션이 없을 때만 신규 진입
            if self.positions[coin_to_act]['size'] == 0:
                pos_type = 1 if action_type == 1 else -1
                leverage = self.leverage_tiers[leverage_idx]
                position_size_ratio = self.position_size_tiers[size_idx]
                invest_amount = self.balance * position_size_ratio
                
                if self.balance >= invest_amount > 0:
                    self.balance -= invest_amount
                    entry_price_with_slippage = current_price * (1 + self.slippage_rate) if pos_type == 1 else current_price * (1 - self.slippage_rate)
                    self.positions[coin_to_act] = {'type': pos_type, 'leverage': leverage, 'entry_price': entry_price_with_slippage, 'size': invest_amount}

        elif action_type == 3: # 포지션 종료
            if self.positions[coin_to_act]['size'] > 0:
                p = self.positions[coin_to_act]
                exit_price_with_slippage = current_price * (1 - self.slippage_rate) if p['type'] == 1 else current_price * (1 + self.slippage_rate)
                
                pnl = (exit_price_with_slippage - p['entry_price']) * p['type'] * (p['size'] / p['entry_price']) * p['leverage']
                pnl -= p['size'] * p['leverage'] * self.trade_fee
                
                self.balance += (p['size'] + pnl)
                self.positions[coin_to_act] = {'type': 0, 'leverage': 0, 'entry_price': 0, 'size': 0}

    def _calculate_portfolio_value(self):
        assets_value = 0
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        for name, p in self.positions.items():
            if p['size'] > 0:
                pnl = (current_prices[name] - p['entry_price']) * p['type'] * (p['size'] / p['entry_price']) * p['leverage']
                assets_value += (p['size'] + pnl)
        return self.balance + assets_value

    def _check_stop_loss(self):
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        for name, p in self.positions.items():
            if p['size'] > 0:
                pnl_ratio = (current_prices[name] - p['entry_price']) / p['entry_price'] * p['type']
                if pnl_ratio < -self.stop_loss_pct:
                    self._execute_trade([self.coin_names.index(name), 3, 0, 0]) # 해당 코인 포지션 종료

# -----------------------------------------------------------------------------
# 3. 학습 실행 (커리큘럼 학습 적용)
# -----------------------------------------------------------------------------
def train_with_curriculum(data_folder, model_path_prefix="ppo_final"):
    device = "auto"
    print(f"'{device}' 장치를 사용하여 커리큘럼 학습을 시작합니다.")

    # ⭐ 커리큘럼 단계 정의
    curriculum_stages = [
        {'stage': 1, 'max_coins': 5,  'leverage_tiers': [1, 2], 'total_timesteps': 500_000},
        {'stage': 2, 'max_coins': 20, 'leverage_tiers': [1, 2, 5], 'total_timesteps': 1_000_000},
        {'stage': 3, 'max_coins': 100,'leverage_tiers': [1, 2, 5, 10], 'total_timesteps': 1_500_000},
    ]
    model = None
    
    for stage_info in curriculum_stages:
        stage = stage_info['stage']
        print(f"\n===== STAGE {stage} 시작: {stage_info} =====\n")

        env = DummyVecEnv([lambda: AdvancedTradingEnv(
            data_folder,
            max_coins=stage_info['max_coins'],
            leverage_tiers=stage_info['leverage_tiers']
        )])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

        if model is None:
            model = PPO(MultiAssetPolicy, env, verbose=1, device=device,
                tensorboard_log=f"./ppo_curriculum_log/stage_{stage}",
                learning_rate=1e-4, n_steps=2048, batch_size=64, n_epochs=10)
        else:
            model.set_env(env)

        model.learn(total_timesteps=stage_info['total_timesteps'], reset_num_timesteps=False, progress_bar=True)
        
        model.save(f"{model_path_prefix}_stage_{stage}.zip")
        env.save(f"vec_normalize_stage_{stage}.pkl")

    print("\n===== 모든 커리큘럼 학습 완료! =====")

if __name__ == "__main__":
    CRYPTO_DATA_FOLDER = "crypto_15m"
    if not os.path.exists(CRYPTO_DATA_FOLDER):
        print(f"오류: '{CRYPTO_DATA_FOLDER}' 폴더를 찾을 수 없습니다.")
    else:
        train_with_curriculum(CRYPTO_DATA_FOLDER)
