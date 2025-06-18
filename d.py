import os
import shutil
import math
import logging
import random
import time
from collections import deque
from datetime import datetime, timedelta
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm
import ccxt
import quantstats
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor


class Config:
   
    SEED = 42
    # 분석할 코인 목록
    TICKERS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 'SHIB/USDT', 'LTC/USDT']
    TIME_INTERVAL = "15m"

    # 데이터 기간 설정
    TRAIN_START_DATE = "2020-01-01"
    TRAIN_END_DATE = "2024-01-01"
    FORWARD_TEST_START_DATE = "2024-01-01"
    FORWARD_TEST_END_DATE = "2025-03-01"

    # 파일 경로
    DATA_DIR = "data"
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data_multi.parquet")
    FORWARD_TEST_DATA_PATH = os.path.join(DATA_DIR, "forward_test_data_multi.parquet")
    
    MODEL_DIR = "models_v4.4_multi"
    LOG_DIR = "logs_v4_4_multi"
    FINAL_REPORT_DIR = "final_report_v4_4_multi"
    
    CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "champion_model.zip")
    CHALLENGER_MODEL_PATH = os.path.join(MODEL_DIR, "challenger_model.zip")
    VECNORM_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")
    TB_LOG_DIR = os.path.join(LOG_DIR, "tensorboard")
    MONITOR_CSV = os.path.join(LOG_DIR, "monitor.csv")

    # 학습 파라미터
    NUM_GENERATIONS = 30
    TOTAL_TRAINING_TIMESTEPS_PER_GEN = 100_000
    PATIENCE_LIMIT = 5
    EVAL_FREQ = 10_000

    # 환경 설정
    WINDOW_SIZE = 72
    INITIAL_BALANCE = 10000.0
    MAX_COINS = len(TICKERS)
    
    # 거래 비용
    TRADE_FEE = 0.001
    BASE_SLIPPAGE_RATE = 0.0005
    SLIPPAGE_ALPHA = 0.1
    SLIPPAGE_BETA = 2.0
    STOP_LOSS_PCT = 0.10

    # 액션 공간 설정
    ACTION_TYPES = 4  # 0:HOLD, 1:LONG, 2:SHORT, 3:CLOSE
    LEVERAGE_TIERS = [1, 2, 5]
    POSITION_SIZE_TIERS = [0.1, 0.25, 0.5]

    # 보상 함수 파라미터
    DRAWDOWN_PENALTY_FACTOR = 2e-3
    VOLATILITY_PENALTY_FACTOR = 1e-2
    EXPOSURE_PENALTY_FACTOR = 5e-4
    DURATION_PENALTY_FACTOR = 1e-5
    OVERWEIGHT_PENALTY_FACTOR = 1e-3
    MAX_COIN_EXPOSURE_PCT = 0.5
    MAX_HOLDING_DURATION = 96 * 7 # 15분봉 * 96 (하루) * 7일
    REWARD_CLIP_RANGE = (-0.05, 0.05)
    
    # PPO 하이퍼파라미터
    REPLAY_BUFFER_SIZE = 20_000
    AUXILIARY_UPDATES = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PPO_PARAMS = {
        'learning_rate': 3e-5,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.995,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'clip_range': 0.2,
    }

# --- 유틸리티 ---
def setup_logging(config: Config):
    """파일 및 콘솔 출력을 위한 로거를 설정합니다."""
    for path in [config.DATA_DIR, config.MODEL_DIR, config.LOG_DIR, config.FINAL_REPORT_DIR]:
        os.makedirs(path, exist_ok=True)
    
    log_filename = os.path.join(config.LOG_DIR, f"training_seed_{config.SEED}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 데이터 준비 ---
# --- 데이터 준비 ---
def prepare_data_binance(config: Config):
    """
    Binance API를 사용하여 학습 및 테스트 데이터를 다운로드하고 Parquet 형식으로 저장합니다.
    """
    exchange = ccxt.binance()
    
    def fetch_and_save(start_date, end_date, filepath, tickers):
        if os.path.exists(filepath):
            try:
                existing_df = pd.read_parquet(filepath)
                if existing_df.shape[0] > config.WINDOW_SIZE:
                    logging.info(f"✅ Valid data file already exists: {filepath} (Shape: {existing_df.shape})")
                    return
                else:
                    logging.warning(f"Existing file {filepath} has insufficient data. Re-downloading...")
                    os.remove(filepath)
            except Exception as e:
                logging.warning(f"Could not read existing file {filepath}: {e}. Re-downloading...")
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        logging.info(f"Fetching data from {start_date} to {end_date} for {filepath}...")
        
        all_dfs = []
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        for ticker in tqdm(tickers, desc=f"Fetching {os.path.basename(filepath)}"):
            try:
                ohlcvs = []
                current_ts = start_ts
                
                while current_ts < end_ts:
                    try:
                        # fetch_ohlcv는 한 번에 제한된 수의 캔들만 가져올 수 있으므로 반복 호출
                        chunk = exchange.fetch_ohlcv(ticker, config.TIME_INTERVAL, since=current_ts, limit=1000)
                        if not chunk:
                            break
                        ohlcvs.extend(chunk)
                        current_ts = chunk[-1][0] + (exchange.parse_timeframe(config.TIME_INTERVAL) * 1000)
                        time.sleep(exchange.rateLimit / 1000) # Rate limit 준수
                    except Exception as chunk_error:
                        logging.error(f"Error fetching chunk for {ticker}: {chunk_error}")
                        time.sleep(1)
                        break

                if not ohlcvs:
                    logging.warning(f"No data found for {ticker} in the given range.")
                    continue
                
                df = pd.DataFrame(ohlcvs, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 중복 제거 및 정렬
                df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                
                # 날짜 범위 필터링
                df = df[
                    (df['timestamp'] >= pd.to_datetime(start_date)) & 
                    (df['timestamp'] < pd.to_datetime(end_date))
                ]
                
                if len(df) == 0:
                    logging.error(f"No data in date range for {ticker}")
                    continue
                
                df.set_index('timestamp', inplace=True)
                
                # 티커 이름을 "BTC-USD"와 같은 형식으로 통일하고 MultiIndex 생성
                coin_name = ticker.replace('/USDT', '-USD')
                df.columns = pd.MultiIndex.from_product([[coin_name], df.columns])
                all_dfs.append(df)
                
                logging.info(f"✅ {ticker}: {len(df)} data points collected")
                
            except Exception as e:
                logging.error(f"Failed to fetch data for {ticker}: {e}")
                continue

        if not all_dfs:
            logging.error("No data could be fetched for any ticker. Aborting.")
            # 빈 파일을 생성하여 오류를 명확히 함
            pd.DataFrame().to_parquet(filepath)
            return

        # 모든 티커의 데이터를 하나의 데이터프레임으로 결합
        logging.info("Combining all ticker data...")
        final_df = pd.concat(all_dfs, axis=1)
        
        # 결측값 처리
        final_df.fillna(method='ffill', inplace=True) # 누락된 값은 이전 값으로 채움
        final_df.dropna(inplace=True) # 맨 앞의 NaN 값 제거
        
        # 최종 검증
        if final_df.shape[0] == 0 or final_df.shape[1] == 0:
            raise RuntimeError(f"❌ Final DataFrame is empty: {final_df.shape}")
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 저장
        final_df.to_parquet(filepath)
        logging.info(f"✅ Data saved to '{filepath}'. Shape: {final_df.shape}")
        logging.info(f"   Columns: {list(final_df.columns.get_level_values(0).unique())}")
        logging.info(f"   Date range: {final_df.index[0]} to {final_df.index[-1]}")
        
        # 검증 로드
        try:
            test_df = pd.read_parquet(filepath)
            logging.info(f"✅ Verification: Successfully loaded {filepath} with shape {test_df.shape}")
        except Exception as e:
            logging.error(f"❌ Verification failed for {filepath}: {e}")

    # 훈련 데이터 생성
    logging.info("=== Preparing Training Data ===")
    fetch_and_save(config.TRAIN_START_DATE, config.TRAIN_END_DATE, config.TRAIN_DATA_PATH, config.TICKERS)
    
    # 순방향 테스트 데이터 생성
    logging.info("=== Preparing Forward Test Data ===")
    fetch_and_save(config.FORWARD_TEST_START_DATE, config.FORWARD_TEST_END_DATE, config.FORWARD_TEST_DATA_PATH, config.TICKERS)
    
    logging.info("=== Data Preparation Complete ===")


# --- 환경 정의 (Gymnasium) ---
class AdvancedTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_path, config: Config, is_training=True):
        super().__init__()
        self.config = config
        self.is_training = is_training
        
        df = pd.read_parquet(data_path)
        # 설정 파일의 Tickers 순서를 기준으로 열 정렬
        self.coin_names = [ticker.replace('/USDT', '-USD') for ticker in config.TICKERS]
        
        # 데이터프레임의 열 순서를 config.TICKERS 순서와 일치시킴
        self.prices = df.reindex(columns=self.coin_names, level=0)
        
        self.data_length = len(self.prices)
        self.dates = self.prices.index
        self.num_coins = len(self.coin_names)
        
        # ACTION: [coin_index, action_type, size_index]
        self.action_space = spaces.MultiDiscrete([
            self.num_coins, 
            config.ACTION_TYPES, 
            len(config.POSITION_SIZE_TIERS)
        ])
        
        # STATE: (num_coins, window_size, features)
        self.num_features = 5 # open, high, low, close, volume
        self.num_portfolio_features = 6 # position_type, pnl, size, leverage, duration, drawdown
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_coins, config.WINDOW_SIZE, self.num_features + self.num_portfolio_features),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = self.config.WINDOW_SIZE
        self.balance = self.config.INITIAL_BALANCE
        self.portfolio_value = self.config.INITIAL_BALANCE
        self.peak_portfolio_value = self.config.INITIAL_BALANCE
        self.positions = {name: {'type': 0, 'leverage': 1, 'entry_price': 0, 'size': 0, 'pnl': 0, 'duration': 0} for name in self.coin_names}
        self.info_history = []
        self.recent_pnl_history = deque(maxlen=100)
        
        return self._get_obs(), self._get_info()

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        window_slice = slice(self.index - self.config.WINDOW_SIZE, self.index)
        
        market_data_slice = self.prices.iloc[window_slice]
        
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value if self.peak_portfolio_value > 0 else 0
        
        for i, coin in enumerate(self.coin_names):
            # Market data - MultiIndex DataFrame 접근 방식 수정
            try:
                # MultiIndex에서 특정 코인의 모든 feature 데이터 가져오기
                coin_data = market_data_slice.xs(coin, level=0, axis=1).values
                
                if coin_data.shape == (self.config.WINDOW_SIZE, self.num_features):
                    obs[i, :, :self.num_features] = coin_data
                else: # 데이터 shape이 맞지 않을 경우 대비
                    logging.warning(f"Data shape mismatch for {coin}. Got {coin_data.shape}, expected {(self.config.WINDOW_SIZE, self.num_features)}. Filling with last known value.")
                    if len(coin_data) > 0:
                        obs[i, :, :self.num_features] = np.tile(coin_data[-1], (self.config.WINDOW_SIZE, 1))
                    else: # 데이터가 아예 없는 경우
                        obs[i, :, :self.num_features] = np.ones((self.config.WINDOW_SIZE, self.num_features)) * 100  # 기본값

            except (KeyError, IndexError) as e:
                logging.warning(f"Data for {coin} not found at index {self.index}: {e}. Using default values.")
                obs[i, :, :self.num_features] = np.ones((self.config.WINDOW_SIZE, self.num_features)) * 100
            
            # Portfolio state
            pos = self.positions[coin]
            pnl_norm = pos['pnl'] / self.config.INITIAL_BALANCE if self.config.INITIAL_BALANCE > 0 else 0
            size_norm = pos['size'] / self.config.INITIAL_BALANCE if self.config.INITIAL_BALANCE > 0 else 0
            duration_norm = pos['duration'] / self.config.MAX_HOLDING_DURATION if self.config.MAX_HOLDING_DURATION > 0 else 0
            
            obs[i, :, 5] = pos['type']
            obs[i, :, 6] = pnl_norm
            obs[i, :, 7] = size_norm
            obs[i, :, 8] = pos['leverage']
            obs[i, :, 9] = duration_norm
            obs[i, :, 10] = current_drawdown
            
        return obs
        
    def _get_info(self):
        info = {
            'date': self.dates[self.index] if self.index < len(self.dates) else self.dates[-1],
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'positions': self.positions.copy()
        }
        # Add individual PnLs to info for easier logging/analysis
        for coin, pos in self.positions.items():
            info[f'pnl_{coin}'] = pos['pnl']
        return info

    def _update_portfolio_value(self):
        assets_in_positions = 0
        unrealized_pnl = 0
        
        # MultiIndex DataFrame에서 가격 접근 방식 수정
        current_prices = {}
        for name in self.coin_names:
            try:
                current_prices[name] = self.prices.xs(name, level=0, axis=1)['close'].iloc[self.index]
            except (KeyError, IndexError):
                current_prices[name] = 100.0  # 기본값
        
        for name, p in self.positions.items():
            if p['size'] > 0:
                price_diff = current_prices[name] - p['entry_price']
                # PNL 계산: (현재가 - 진입가) * (포지션 수량) * 레버리지 * 타입(1: long, -1: short)
                # 포지션 수량 = 투자금액 / 진입가
                pnl = price_diff * (p['size'] / p['entry_price']) * p['leverage'] * p['type']
                self.positions[name]['pnl'] = pnl
                assets_in_positions += p['size'] 
                unrealized_pnl += pnl
                self.positions[name]['duration'] += 1

        self.portfolio_value = self.balance + assets_in_positions + unrealized_pnl
        self.peak_portfolio_value = max(self.peak_portfolio_value, self.portfolio_value)

    def _execute_trade(self, action):
        coin_idx, action_type, size_idx = action[0], action[1], action[2]
        
        if action_type == 0: # Hold
            return

        coin_to_act = self.coin_names[coin_idx]
        current_price = self.prices.loc[self.prices.index[self.index], (coin_to_act, 'close')]

        if action_type == 3: # Close position
            self._close_position(coin_to_act, current_price)
        elif action_type in [1, 2]: # Open a new position
            if self.positions[coin_to_act]['size'] > 0: # 이미 포지션이 있으면 추가 진입 불가
                return

            leverage = self.config.LEVERAGE_TIERS[0] # 간단하게 첫번째 레버리지 사용
            size_ratio = self.config.POSITION_SIZE_TIERS[size_idx]
            investment_amount = self.balance * size_ratio
            
            if self.balance >= investment_amount and investment_amount > 0:
                pos_type = 1 if action_type == 1 else -1
                notional_value = investment_amount * leverage
                slip = self.config.BASE_SLIPPAGE_RATE * (notional_value / 10000)**self.config.SLIPPAGE_BETA
                entry_price = current_price * (1 + slip * pos_type)
                fee = notional_value * self.config.TRADE_FEE
                
                self.balance -= (investment_amount + fee)
                self.positions[coin_to_act] = {
                    'type': pos_type,
                    'leverage': leverage,
                    'entry_price': entry_price,
                    'size': investment_amount, # 투자 원금
                    'pnl': -fee, # 초기 PnL은 수수료
                    'duration': 1
                }
                # logging.info(f"Time: {self.dates[self.index]}, Action: {'LONG' if pos_type==1 else 'SHORT'} {coin_to_act}, Size: {investment_amount:.2f}, Price: {current_price:.2f}")

    def _close_position(self, coin_name, exit_price):
        p = self.positions[coin_name]
        if p['size'] == 0:
            return

        slip = self.config.BASE_SLIPPAGE_RATE * ((p['size'] * p['leverage'])**self.config.SLIPPAGE_BETA)
        exit_price_with_slip = exit_price * (1 - slip * p['type'])
        
        price_diff = exit_price_with_slip - p['entry_price']
        pnl = price_diff * (p['size'] / p['entry_price']) * p['leverage'] * p['type']

        fee = (p['size'] * p['leverage']) * self.config.TRADE_FEE # 진입, 청산 수수료
        
        self.balance += p['size'] + pnl - fee
        # logging.info(f"Time: {self.dates[self.index]}, Action: CLOSE {coin_name}, PnL: {pnl - fee:.2f}, Balance: {self.balance:.2f}")
        self.positions[coin_name] = {'type': 0, 'leverage': 1, 'entry_price': 0, 'size': 0, 'pnl': 0, 'duration': 0}


    def _check_stop_loss(self):
        current_prices = {name: self.prices[name]['close'].iloc[self.index] for name in self.coin_names}
        for name, p in self.positions.copy().items():
            if p['size'] > 0:
                pnl_ratio = (current_prices[name] - p['entry_price']) / p['entry_price'] * p['type'] * p['leverage']
                if pnl_ratio < -self.config.STOP_LOSS_PCT:
                    # logging.info(f"Stop loss triggered for {name} at price {current_prices[name]}")
                    self._close_position(name, current_prices[name])

    def _calculate_reward(self, prev_portfolio_value):
        pnl = self.portfolio_value - prev_portfolio_value
        # pnl을 포트폴리오 가치로 나눈 수익률을 보상으로 사용
        reward_pnl = pnl / prev_portfolio_value if prev_portfolio_value != 0 else 0
        clipped_pnl = np.clip(reward_pnl, *self.config.REWARD_CLIP_RANGE)
        
        drawdown = max(0, (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value)
        drawdown_penalty = self.config.DRAWDOWN_PENALTY_FACTOR * drawdown

        total_exposure = sum(p['size'] * p['leverage'] for p in self.positions.values())
        exposure_ratio = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        exposure_penalty = self.config.EXPOSURE_PENALTY_FACTOR * max(0, exposure_ratio - 1.0)
        
        duration_penalty = 0
        overweight_penalty = 0
        for name, p in self.positions.items():
            if p['size'] > 0:
                if p['duration'] > self.config.MAX_HOLDING_DURATION:
                    duration_penalty += self.config.DURATION_PENALTY_FACTOR
                coin_exposure_pct = (p['size'] * p['leverage']) / self.portfolio_value if self.portfolio_value > 0 else 0
                if coin_exposure_pct > self.config.MAX_COIN_EXPOSURE_PCT:
                    overweight_penalty += self.config.OVERWEIGHT_PENALTY_FACTOR
        
        reward = clipped_pnl - drawdown_penalty - exposure_penalty - duration_penalty - overweight_penalty
        return reward

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        
        self._execute_trade(action)
        self.index += 1 # 가격을 먼저 업데이트 하고 포트폴리오를 계산
        
        terminated = (self.index >= self.data_length - 1) or (self.portfolio_value < self.config.INITIAL_BALANCE * 0.2)
        if terminated:
            # 에피소드 종료 시 모든 포지션 정리
            current_prices = {name: self.prices[name]['close'].iloc[self.index-1] for name in self.coin_names}
            for coin in self.coin_names:
                self._close_position(coin, current_prices[coin])
            
            obs = self._get_obs()
            info = self._get_info()
            reward = self._calculate_reward(prev_portfolio_value)
            return obs, reward, terminated, False, info
        
        self._update_portfolio_value()
        self._check_stop_loss()
        
        reward = self._calculate_reward(prev_portfolio_value)
        truncated = False
        
        obs = self._get_obs()
        info = self._get_info()
        info['reward'] = reward
        self.info_history.append(info)
        
        return obs, reward, terminated, truncated, info
    
    def get_info_df(self):
        return pd.DataFrame(self.info_history).set_index('date')

# --- 신경망 아키텍처 ---
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]
        self.num_coins = observation_space.shape[0]

        self.cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            ) for _ in range(self.num_coins)
        ])
        
        with torch.no_grad():
            dummy_input = torch.as_tensor(observation_space.sample()[None]).float()
            dummy_coin_input = dummy_input[:, 0, :, :].permute(0, 2, 1)
            n_flatten = self.cnns[0](dummy_coin_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(self.num_coins * n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded_features = []
        for i in range(self.num_coins):
            coin_data = observations[:, i, :, :].permute(0, 2, 1) 
            encoded_features.append(self.cnns[i](coin_data))
        
        concatenated_features = torch.cat(encoded_features, dim=1)
        return self.linear(concatenated_features)

# --- Main Execution Block ---
def main():
    config = Config()
    setup_logging(config)
    set_seed(config.SEED)
    
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", help="Execution mode: prepare_data, train, evaluate", default="train")
    args = parser.parse_args()
    
    if args.mode == "prepare_data":
        prepare_data_binance(config)
        return

    if args.mode == "train":
        if not os.path.exists(config.TRAIN_DATA_PATH) or not os.path.exists(config.FORWARD_TEST_DATA_PATH):
            logging.info("Data not found. Preparing data first...")
            prepare_data_binance(config)
        
        env_lambda = lambda: Monitor(AdvancedTradingEnv(data_path=config.TRAIN_DATA_PATH, config=config, is_training=True), config.MONITOR_CSV)
        train_env = DummyVecEnv([env_lambda])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=config.PPO_PARAMS['gamma'])

        policy_kwargs = dict(
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )

        model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log=config.TB_LOG_DIR, policy_kwargs=policy_kwargs, **config.PPO_PARAMS)
        
        champion_score = -float('inf')
        patience_counter = 0

        for generation in range(1, config.NUM_GENERATIONS + 1):
            logging.info(f"\n{'='*25} GENERATION {generation}/{config.NUM_GENERATIONS} {'='*25}")
            
            model.learn(total_timesteps=config.TOTAL_TRAINING_TIMESTEPS_PER_GEN, reset_num_timesteps=False, tb_log_name=f"PPO_Gen_{generation}")

            logging.info("--- Evaluating Challenger ---")
            train_env.save(config.VECNORM_PATH) # 평가 전에 VecNormalize 상태 저장
            eval_score = run_quick_backtest(config, model, config.VECNORM_PATH, config.FORWARD_TEST_DATA_PATH)
            logging.info(f"Generation {generation} Score: ${eval_score:,.2f}")

            if eval_score > champion_score:
                logging.info(f"🏆 New Champion! Score improved from ${champion_score:,.2f} to ${eval_score:,.2f}")
                champion_score = eval_score
                model.save(config.CHAMPION_MODEL_PATH)
                train_env.save(config.VECNORM_PATH) # 챔피언 모델과 함께 VecNormalize 상태 저장
                patience_counter = 0
            else:
                patience_counter += 1
                logging.info(f"🛡️ Champion defends the title. Patience: {patience_counter}/{config.PATIENCE_LIMIT}")

            if patience_counter >= config.PATIENCE_LIMIT:
                logging.info("🛑 Early stopping due to lack of improvement.")
                break
        
        logging.info("Training complete. Running final analysis...")
        analyze_final_champion(config)

    elif args.mode == "evaluate":
        analyze_final_champion(config)
    else:
        raise ValueError("Invalid mode. Choose from 'prepare_data', 'train', or 'evaluate'.")


def run_quick_backtest(config: Config, model, vecnorm_path: str, data_path: str) -> float:
    try:
        eval_env_lambda = lambda: AdvancedTradingEnv(data_path=data_path, config=config, is_training=False)
        eval_env = DummyVecEnv([eval_env_lambda])
        
        if os.path.exists(vecnorm_path):
             eval_env = VecNormalize.load(vecnorm_path, eval_env)
        
        eval_env.training = False
        eval_env.norm_reward = False

        obs = eval_env.reset()
        done = False
        infos = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, info_list = eval_env.step(action)
            done = dones[0]
            infos.append(info_list[0])
            
        # 키 이름을 'portfolio_value'로 수정
        final_value = infos[-1].get('portfolio_value', 0.0)
        return final_value

    except Exception as e:
        logging.error(f"Quick backtest failed: {e}", exc_info=True)
        return -float('inf')


def analyze_final_champion(config: Config):
    logging.info(f"\n{'='*25} FINAL CHAMPION ANALYSIS {'='*25}")
    model_path = config.CHAMPION_MODEL_PATH
    data_path = config.FORWARD_TEST_DATA_PATH
    report_path_base = os.path.join(config.FINAL_REPORT_DIR, f'final_champion_seed{config.SEED}')

    if not os.path.exists(model_path):
        logging.error(f"Error: Champion model not found at {model_path}.")
        return

    if not os.path.exists(config.VECNORM_PATH):
        logging.error(f"Error: VecNormalize stats not found at {config.VECNORM_PATH}.")
        return

    eval_env_lambda = lambda: AdvancedTradingEnv(data_path=data_path, config=config, is_training=False)
    raw_env = DummyVecEnv([eval_env_lambda])
    env = VecNormalize.load(config.VECNORM_PATH, raw_env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(model_path, env=env)

    logging.info("Running detailed backtest for analysis...")
    all_infos = []
    obs = env.reset()
    done = False
    
    # tqdm의 total 값 계산 수정
    raw_env_instance = raw_env.envs[0]
    pbar_total = raw_env_instance.data_length - raw_env_instance.config.WINDOW_SIZE - 1
    pbar = tqdm(total=pbar_total, desc="Final Analysis Backtest")
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        all_infos.append(infos[0])
        done = dones[0]
        pbar.update(1)
    pbar.close()

    if not all_infos:
        logging.warning("⚠️ Backtesting resulted in no data. Cannot generate report.")
        return
        
    results_df = pd.DataFrame(all_infos)
    os.makedirs(config.FINAL_REPORT_DIR, exist_ok=True)
    results_df.to_csv(f"{report_path_base}_results.csv", index=False)
    
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df.set_index('date', inplace=True)
    
    # 키 이름을 'portfolio_value'로 수정
    returns_series = results_df['portfolio_value'].pct_change().fillna(0)
    
    logging.info("Generating QuantStats report...")
    try:
        # 보고서 제목 수정
        quantstats.reports.html(returns_series, output=f"{report_path_base}_quantstats.html", title=f'Final Champion (V4.4 Multi-Asset Seed {config.SEED})')
        logging.info(f"✅ QuantStats report saved to {report_path_base}_quantstats.html")
    except Exception as e:
        logging.error(f"❌ Could not generate QuantStats report: {e}")

# 정상적인 main 함수 실행 구문
if __name__ == "__main__":
    main()
