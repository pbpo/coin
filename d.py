import os
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 멀티-에셋 트랜스포머 정책 (Multi-Asset Transformer Policy)
# -----------------------------------------------------------------------------
# 설명: (코인 수, 윈도우 크기, 특징 수) 형태의 3D 데이터를 처리하기 위한 특징 추출기입니다.
# 각 코인의 시계열 데이터를 개별적으로 처리한 후, 모든 정보를 종합하여 최종 특징을 만듭니다.
class MultiAssetTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, nhead=8, num_layers=2):
        num_coins = observation_space.shape[0]
        # 최종 특징 차원: 코인 수 * d_model
        features_dim = num_coins * d_model
        super().__init__(observation_space, features_dim=features_dim)

        self.d_model = d_model
        self.num_coins = num_coins
        _, seq_len, feat_dim = observation_space.shape
        
        self.input_proj = nn.Linear(feat_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x의 shape: (batch_size, num_coins, seq_len, feat_dim)
        batch_size = x.shape[0]
        
        # 모든 코인 데이터를 (batch_size * num_coins, seq_len, feat_dim) 형태로 변환
        x_reshaped = x.view(batch_size * self.num_coins, x.shape[2], x.shape[3])
        
        projected = self.input_proj(x_reshaped)
        encoded = self.transformer(projected)
        
        # 각 시퀀스의 마지막 타임스텝 결과만 사용 (최신 정보를 요약)
        last_step_encoded = encoded[:, -1, :]
        
        # 다시 코인별로 분리된 형태로 복원
        output = last_step_encoded.reshape(batch_size, self.num_coins * self.d_model)
        return output

class MultiAssetPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=MultiAssetTransformerExtractor,
            # 고성능 GPU를 고려하여 모델 파라미터 상향 조정
            features_extractor_kwargs=dict(d_model=128, nhead=8, num_layers=4),
        )

# -----------------------------------------------------------------------------
# 2. 대규모 트레이딩 환경 (Large-Scale Trading Environment)
# -----------------------------------------------------------------------------
class LargeScaleTradingEnv(gym.Env):
    def __init__(self, data_folder, window_size=64, trade_fee=0.001, max_coins=100):
        super().__init__()
        
        # --- 1. 데이터 로드 및 전처리 (메모리 내에서 모두 처리) ---
        print(f"'{data_folder}'에서 최대 {max_coins}개의 코인 데이터를 로드합니다...")
        all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
        df_list = []
        for file in tqdm(all_files[:max_coins], desc="파일 로딩 중"):
            coin_name = os.path.basename(file).split('_')[0]
            df = pd.read_csv(file, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'], parse_dates=['timestamp'])
            df['coin'] = coin_name
            df_list.append(df)

        print("데이터 동기화를 위해 피벗 테이블을 생성합니다 (RAM 사용량이 높을 수 있습니다)...")
        full_df = pd.concat(df_list, ignore_index=True)
        pivot_df = full_df.pivot_table(index='timestamp', columns='coin', values=['open', 'high', 'low', 'close', 'volume'])
        
        print("결측값을 채웁니다...")
        pivot_df.ffill(inplace=True)
        pivot_df.bfill(inplace=True)
        pivot_df.dropna(inplace=True) # 맨 앞/뒤에 남은 NaN 제거
        
        self.coin_names = sorted(pivot_df.columns.get_level_values(1).unique().tolist())
        self.num_coins = len(self.coin_names)
        
        # 데이터 형태 변환 (Timestamps, Features, Coins) -> (Coins, Timestamps, Features) 및 정규화
        prices_raw = pivot_df.values.reshape(len(pivot_df), 5, self.num_coins)
        self.prices = np.transpose(prices_raw, (2, 0, 1))
        
        mean = self.prices.mean(axis=1, keepdims=True)
        std = self.prices.std(axis=1, keepdims=True)
        self.normalized_prices = (self.prices - mean) / (std + 1e-8)
        
        self.window_size = window_size
        self.trade_fee = trade_fee
        print(f"총 {self.num_coins}개 코인, {len(pivot_df)} 타임스텝 데이터 준비 완료.")

        # --- 2. 행동 공간 정의 (계층적 결정) ---
        # 0: 관망, 1: 최적 코인 롱(2x), 2: 최적 코인 롱(5x), 3: 최적 코인 숏(2x), 4: 최적 코인 숏(5x), 5: 모든 포지션 종료
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_coins, window_size, 5), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.index = self.window_size
        self.positions = {name: {'type': 0, 'leverage': 0, 'entry_price': 0} for name in self.coin_names}
        self.realized_pnl = 0
        return self._get_obs()

    def _get_obs(self):
        return self.normalized_prices[:, self.index - self.window_size : self.index, :]

    def _find_best_coin_for_action(self, pos_type):
        """간단한 모멘텀(최근 가격 변화율) 기반으로 최적의 코인을 찾는 함수"""
        best_coin = None
        best_momentum = -np.inf

        # 모멘텀 계산 (window_size의 절반 기간 동안의 가격 변화율)
        momentum_window = self.window_size // 2
        price_data = self.prices[:, self.index - momentum_window : self.index, 3] # 종가
        momentum = (price_data[:, -1] - price_data[:, 0]) / price_data[:, 0]

        candidate_momentum = momentum * pos_type # 롱이면 양수, 숏이면 음수 모멘텀
        
        for i in range(self.num_coins):
            coin_name = self.coin_names[i]
            if self.positions[coin_name]['type'] == 0: # 포지션이 없는 코인 중에서
                if candidate_momentum[i] > best_momentum:
                    best_momentum = candidate_momentum[i]
                    best_coin = coin_name
        return best_coin

    def step(self, action):
        reward = 0
        done = self.index >= self.prices.shape[1] - 1
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}

        if 1 <= action <= 4:
            pos_type, leverage = {1: (1, 2), 2: (1, 5), 3: (-1, 2), 4: (-1, 5)}[action]
            best_coin = self._find_best_coin_for_action(pos_type)
            if best_coin:
                self._open_position(best_coin, pos_type, leverage, current_prices[best_coin])
        
        elif action == 5:
            for name in self.coin_names:
                if self.positions[name]['type'] != 0:
                    reward += self._close_position(name, current_prices[name])

        unrealized_pnl = 0
        for name, p in self.positions.items():
            if p['type'] != 0:
                pnl_ratio = (current_prices[name] - p['entry_price']) / p['entry_price']
                unrealized_pnl += pnl_ratio * p['type'] * p['leverage']

        final_reward = reward + (unrealized_pnl / self.num_coins if self.num_coins > 0 else 0)
        self.index += 1
        return self._get_obs(), final_reward, done, {}

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
# 3. 학습 실행 (Training Runner)
# -----------------------------------------------------------------------------
def train_on_gpu(data_folder, model_path="ppo_large_gpu.zip"):
    # --- GPU 설정 ---
    # AMD GPU(ROCm) 또는 NVIDIA(CUDA)를 자동으로 감지합니다.
    # PyTorch가 GPU를 인식하지 못하면 CPU로 자동 전환됩니다.
    device = "auto" 
    
    print(f"'{device}' 장치를 사용하여 학습을 시작합니다.")
    print("환경을 설정합니다...")
    env = DummyVecEnv([lambda: LargeScaleTradingEnv(data_folder)])
    
    # 모델 파라미터는 Stable Baselines3 문서 참고 (https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
    model = PPO(
        MultiAssetPolicy, 
        env, 
        verbose=1, 
        device=device,
        tensorboard_log="./ppo_gpu_log",
        n_steps=2048,           # 업데이트까지 더 많은 샘플 수집
        batch_size=64,         # GPU 메모리에 맞게 배치 크기 조정
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01           # 탐험을 장려하기 위한 엔트로피 계수
    )
    
    print("학습을 시작합니다 (GPU 사용)...")
    # 대규모 데이터셋이므로 총 타임스텝을 늘려 충분히 학습시킵니다.
    model.learn(total_timesteps=2_000_000, progress_bar=True)
    
    print(f"학습 완료! 모델을 {model_path}에 저장합니다.")
    model.save(model_path)


if __name__ == "__main__":
    CRYPTO_DATA_FOLDER = "crypto_15m" # 데이터 폴더 경로
    
    if not os.path.exists(CRYPTO_DATA_FOLDER):
        print(f"오류: '{CRYPTO_DATA_FOLDER}' 폴더를 찾을 수 없습니다.")
        print("스크립트와 같은 위치에 데이터 폴더를 생성하고 CSV 파일들을 넣어주세요.")
    else:
        train_on_gpu(CRYPTO_DATA_FOLDER)
