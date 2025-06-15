import os
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 멀티-에셋 트랜스포머 정책 (Positional Encoding 추가)
# -----------------------------------------------------------------------------

# ⭐ 수정포인트 2: Positional Encoding 클래스 추가 (제안 내용 반영)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512): # window_size를 고려하여 max_len 조정 가능
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiAssetTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, nhead=8, num_layers=2):
        num_coins = observation_space.shape[0]
        features_dim = num_coins * d_model
        super().__init__(observation_space, features_dim=features_dim)

        self.d_model = d_model
        self.num_coins = num_coins
        _, seq_len, feat_dim = observation_space.shape
        
        self.input_proj = nn.Linear(feat_dim, d_model)
        # ⭐ 수정포인트 2: PositionalEncoding 적용
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size * self.num_coins, x.shape[2], x.shape[3])
        
        projected = self.input_proj(x_reshaped)
        # ⭐ 수정포인트 2: PositionalEncoding 레이어 통과
        projected_with_pos = self.pos_encoder(projected)
        encoded = self.transformer(projected_with_pos)
        
        last_step_encoded = encoded[:, -1, :]
        output = last_step_encoded.reshape(batch_size, self.num_coins * self.d_model)
        return output

class MultiAssetPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            features_extractor_class=MultiAssetTransformerExtractor,
            features_extractor_kwargs=dict(d_model=128, nhead=8, num_layers=4),
        )

# -----------------------------------------------------------------------------
# 2. 대규모 트레이딩 환경 (Sharpe Ratio 보상 적용)
# -----------------------------------------------------------------------------
class LargeScaleTradingEnv(gym.Env):
    def __init__(self, data_folder, window_size=64, trade_fee=0.001, max_coins=100, reward_window=252):
        super().__init__()
        
        # --- 1. 데이터 로드 (기존과 동일) ---
        # (데이터 로드 코드는 생략)
        print(f"'{data_folder}'에서 최대 {max_coins}개의 코인 데이터를 로드합니다...")
        all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
        df_list = []
        for file in tqdm(all_files[:max_coins], desc="파일 로딩 중"):
            coin_name = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['timestamp'])
            df['coin'] = coin_name
            df_list.append(df)

        print("데이터 동기화를 위해 피벗 테이블을 생성합니다...")
        full_df = pd.concat(df_list, ignore_index=True)
        pivot_df = full_df.pivot_table(index='timestamp', columns='coin', values=['open', 'high', 'low', 'close', 'volume'])
        
        print("결측값을 채웁니다...")
        pivot_df.ffill(inplace=True)
        pivot_df.bfill(inplace=True)
        pivot_df.dropna(inplace=True)
        
        self.coin_names = sorted(pivot_df.columns.get_level_values(1).unique().tolist())
        self.num_coins = len(self.coin_names)
        
        prices_raw = pivot_df.values.reshape(len(pivot_df), 5, self.num_coins)
        self.prices = np.transpose(prices_raw, (2, 0, 1))
        
        # 정규화는 VecNormalize에 맡기므로 원본 데이터 사용
        self.raw_obs_prices = self.prices 
        
        self.window_size = window_size
        self.trade_fee = trade_fee
        
        # ⭐ 수정포인트 1: Sharpe Ratio 계산을 위한 변수 추가
        self.reward_window = reward_window # 샤프지수 계산 기간
        self.portfolio_returns_history = deque(maxlen=self.reward_window)
        self.portfolio_value = 1.0 # 초기 포트폴리오 가치를 1로 설정

        print(f"총 {self.num_coins}개 코인, {len(pivot_df)} 타임스텝 데이터 준비 완료.")

        # --- 2. 행동 및 관측 공간 정의 ---
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_coins, window_size, 5), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.index = self.window_size
        self.positions = {name: {'type': 0, 'leverage': 0, 'entry_price': 0} for name in self.coin_names}
        # ⭐ 수정포인트 1: 리셋 시 포트폴리오 관련 변수 초기화
        self.portfolio_value = 1.0
        self.portfolio_returns_history.clear()
        return self._get_obs()

    def _get_obs(self):
        # VecNormalize가 정규화를 수행하므로 원본 데이터를 전달
        return self.raw_obs_prices[:, self.index - self.window_size : self.index, :]

    def _get_current_portfolio_value(self, current_prices):
        # 현재 포지션의 가치를 계산 (단순화를 위해 현금은 무시하고 레버리지 적용된 자산 가치만 고려)
        value = 0
        for name, p in self.positions.items():
            if p['type'] != 0:
                # 진입 가격 대비 현재 가격 비율로 가치 계산
                value += p['leverage'] * (current_prices[name] / p['entry_price'])
        # 포지션이 없다면 기본 가치 1.0 유지
        return value if value > 0 else 1.0

    def step(self, action):
        done = self.index >= self.prices.shape[1] - 1
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        
        # ⭐ 수정포인트 1: 보상 계산 로직 변경 시작
        previous_portfolio_value = self._get_current_portfolio_value(current_prices)
        
        realized_pnl = 0
        # 행동 수행
        if 1 <= action <= 4:
            pos_type, leverage = {1: (1, 2), 2: (1, 5), 3: (-1, 2), 4: (-1, 5)}[action]
            # (코인 선택 로직은 생략, 기존 로직 사용)
            best_coin = self._find_best_coin_for_action(pos_type)
            if best_coin:
                self._open_position(best_coin, pos_type, leverage, current_prices[best_coin])
                previous_portfolio_value -= self.trade_fee * leverage # 거래비용은 자산가치에서 즉시 차감

        elif action == 5:
            for name in self.coin_names:
                if self.positions[name]['type'] != 0:
                    realized_pnl += self._close_position(name, current_prices[name])
            previous_portfolio_value += realized_pnl

        # 시간 한 스텝 이동
        self.index += 1
        next_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        
        # 포트폴리오 가치 계산 및 수익률 기록
        current_portfolio_value = self._get_current_portfolio_value(next_prices)
        step_return = (current_portfolio_value / previous_portfolio_value) - 1.0
        self.portfolio_returns_history.append(step_return)
        self.portfolio_value = current_portfolio_value

        # Sharpe Ratio 보상 계산
        if len(self.portfolio_returns_history) < self.reward_window:
            final_reward = 0 # 히스토리가 충분히 쌓이기 전까지는 보상 없음
        else:
            returns = np.array(self.portfolio_returns_history)
            # 연율화 된 샤프 지수 (일일 수익률 기준, 무위험 수익률 0 가정)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8)
            final_reward = sharpe_ratio

        return self._get_obs(), final_reward, done, {}
    
    # _find_best_coin_for_action, _open_position, _close_position 메소드는 기존과 동일 (생략)
    def _find_best_coin_for_action(self, pos_type):
        """간단한 모멘텀(최근 가격 변화율) 기반으로 최적의 코인을 찾는 함수"""
        best_coin = None
        best_momentum = -np.inf
        momentum_window = self.window_size // 2
        price_data = self.prices[:, self.index - momentum_window : self.index, 3]
        momentum = (price_data[:, -1] - price_data[:, 0]) / price_data[:, 0]
        candidate_momentum = momentum * pos_type
        for i in range(self.num_coins):
            coin_name = self.coin_names[i]
            if self.positions[coin_name]['type'] == 0:
                if candidate_momentum[i] > best_momentum:
                    best_momentum = candidate_momentum[i]
                    best_coin = coin_name
        return best_coin
    
    def _open_position(self, coin, pos_type, leverage, price):
        pos = self.positions[coin]
        pos['type'] = pos_type
        pos['leverage'] = leverage
        pos['entry_price'] = price
    
    def _close_position(self, coin, price):
        pos = self.positions[coin]
        pnl_ratio = (price - pos['entry_price']) / pos['entry_price']
        profit = pos['type'] * pos['leverage'] * pnl_ratio - self.trade_fee * pos['leverage']
        self.positions[coin] = {'type': 0, 'leverage': 0, 'entry_price': 0}
        return profit

# -----------------------------------------------------------------------------
# 3. 학습 실행 (VecNormalize 적용)
# -----------------------------------------------------------------------------
def train_on_gpu(data_folder, model_path="ppo_sharpe.zip", stats_path="vec_normalize_sharpe.pkl"):
    device = "auto"
    print(f"'{device}' 장치를 사용하여 학습을 시작합니다.")
    print("환경을 설정합니다...")
    
    env = DummyVecEnv([lambda: LargeScaleTradingEnv(data_folder)])
    # ⭐ 수정포인트 3: VecNormalize 래퍼 적용 (제안 내용 반영)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        MultiAssetPolicy, env, verbose=1, device=device,
        tensorboard_log="./ppo_sharpe_log",
        n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95,
        # 하이퍼파라미터 소폭 조정
        learning_rate=1e-4, # 학습률을 낮춰 안정성 증가
        ent_coef=0.001,      # 엔트로피를 낮춰 조기 수렴 방지
        clip_range=0.2
    )
    
    print("학습을 시작합니다 (GPU 사용)...")
    model.learn(total_timesteps=2_000_000, progress_bar=True)
    
    print(f"학습 완료! 모델을 {model_path}에 저장합니다.")
    model.save(model_path)
    # ⭐ 수정포인트 3: VecNormalize 통계 저장
    env.save(stats_path)


if __name__ == "__main__":
    CRYPTO_DATA_FOLDER = "crypto_15m"
    if not os.path.exists(CRYPTO_DATA_FOLDER):
        print(f"오류: '{CRYPTO_DATA_FOLDER}' 폴더를 찾을 수 없습니다.")
    else:
        train_on_gpu(CRYPTO_DATA_FOLDER)
