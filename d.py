import os
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import yfinance as yf

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import quantstats

# --- 1. CONFIGURATION ---
class Config:
    # --- File Paths & Data ---
    DATA_SOURCE = "yahoofinance"
    # Select your tickers
    TICKERS = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD',
               'DOGE-USD', 'AVAX-USD', 'LINK-USD', 'MATIC-USD', 'TRX-USD',
               'DOT-USD', 'LTC-USD', 'BCH-USD', 'SHIB-USD', 'LEO-USD',
               'OKB-USD', 'XLM-USD', 'ATOM-USD', 'TUSD-USD', 'HBAR-USD']
    DATA_START_DATE = "2018-01-01"
    DATA_END_DATE = "2024-06-01"
    TIME_INTERVAL = "1D"
    PARQUET_DATA_PATH = "all_coins_ohlcv.parquet"

    # --- Environment ---
    WINDOW_SIZE = 64
    INITIAL_BALANCE = 10000.0
    TRADE_FEE = 0.001
    SLIPPAGE_RATE = 0.0005
    STOP_LOSS_PCT = 0.10 # 10% stop-loss
    HOLDING_BONUS_FACTOR = 1e-5 # Small reward for holding a position
    RISK_PENALTY_FACTOR = 1e-4 # Penalty for high leverage/size

    # --- Model & Training ---
    MODEL_PATH_PREFIX = "ppo_transformer_model"
    LOG_DIR = "./log_dir"
    DEVICE = "auto"

    # --- Curriculum Stages ---
    CURRICULUM_STAGES = [
        {'stage': 1, 'max_coins': 1, 'leverage_tiers': [1], 'position_size_tiers': [0.1, 0.25, 0.5], 'trade_fee': 0.0, 'slippage_rate': 0.0, 'total_timesteps': 200_000},
        {'stage': 2, 'max_coins': 5, 'leverage_tiers': [1, 2, 5], 'position_size_tiers': [0.1, 0.25, 0.5], 'trade_fee': 0.001, 'slippage_rate': 0.0005, 'total_timesteps': 500_000},
        {'stage': 3, 'max_coins': 20, 'leverage_tiers': [1, 2, 5, 10], 'position_size_tiers': [0.1, 0.25, 0.5, 0.75], 'trade_fee': 0.001, 'slippage_rate': 0.0005, 'total_timesteps': 1_500_000},
    ]

    # --- PPO Hyperparameters ---
    PPO_PARAMS = {
        'learning_rate': 3e-5,
        'n_steps': 2048,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.995,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'clip_range': 0.2,
        'gae_lambda': 0.95
    }

# --- 2. DATA PREPARATION UTILITY ---
def prepare_data(config: Config):
    """
    Downloads data using yfinance and saves it in the required parquet format.
    """
    print("Downloading financial data...")
    df = yf.download(
        tickers=config.TICKERS,
        start=config.DATA_START_DATE,
        end=config.DATA_END_DATE,
        interval=config.TIME_INTERVAL,
        group_by='ticker'
    )
    df = df.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
    df.index = pd.to_datetime(df.index)

    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})
    pivot_df = df.pivot(columns='Ticker', values=['open', 'high', 'low', 'close', 'volume'])
    pivot_df.fillna(method='ffill', inplace=True)
    pivot_df.fillna(method='bfill', inplace=True)

    pivot_df.to_parquet(config.PARQUET_DATA_PATH)
    print(f"Data prepared and saved to '{config.PARQUET_DATA_PATH}'")
    print("Data structure (first 5 rows):")
    print(pivot_df.head())


# --- 3. ADVANCED TRADING ENVIRONMENT ---
class AdvancedTradingEnv(gym.Env):
    """
    A custom Gymnasium environment for multi-asset crypto trading.
    """
    def __init__(self, data_path, mode='parquet', window_size=64, max_coins=20,
                 initial_balance=10000.0, leverage_tiers=[1], position_size_tiers=[0.1],
                 trade_fee=0.001, slippage_rate=0.0005, stop_loss_pct=0.1,
                 holding_bonus_factor=1e-5, risk_penalty_factor=1e-4):
        super().__init__()

        if mode == 'parquet':
            pivot_df = pd.read_parquet(data_path)
        else:
            raise ValueError("Unsupported data mode. Use 'parquet'.")

        self.coin_names = sorted(pivot_df.columns.get_level_values(1).unique().tolist())
        if max_coins < len(self.coin_names):
            self.coin_names = self.coin_names[:max_coins]
            pivot_df = pivot_df.loc[:, (slice(None), self.coin_names)]

        self.num_coins = len(self.coin_names)
        num_features = 5 # OHLCV
        prices_raw = pivot_df.values.reshape(len(pivot_df), num_features, self.num_coins)
        self.prices = np.transpose(prices_raw, (2, 0, 1))
        self.data_length = self.prices.shape[1]
        self.pivot_df_index = pivot_df.index # Store index for date lookup

        self.window_size = window_size
        self.initial_balance = initial_balance
        self.leverage_tiers = leverage_tiers
        self.position_size_tiers = position_size_tiers
        self.trade_fee = trade_fee
        self.slippage_rate = slippage_rate
        self.stop_loss_pct = stop_loss_pct
        self.holding_bonus_factor = holding_bonus_factor
        self.risk_penalty_factor = risk_penalty_factor

        self.action_space = gym.spaces.MultiDiscrete([self.num_coins, 4, len(self.leverage_tiers), len(self.position_size_tiers)])

        self.num_portfolio_features = 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.num_coins, window_size, num_features + self.num_portfolio_features),
            dtype=np.float32
        )
        self.info_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.initial_balance
        self.positions = {name: {'type': 0, 'leverage': 0, 'entry_price': 0, 'size': 0, 'pnl': 0} for name in self.coin_names}
        self.info_history = []
        return self._get_obs(), {}
    def get_info_df(self):
        """
        Returns the recorded info_history as a pandas.DataFrame.
        """
        return pd.DataFrame(self.info_history)
    def _get_obs(self):
        market_obs = self.prices[:, self.index - self.window_size:self.index, :]
        portfolio_obs = np.zeros((self.num_coins, self.window_size, self.num_portfolio_features), dtype=np.float32)

        for i, name in enumerate(self.coin_names):
            pos = self.positions[name]
            portfolio_obs[i, :, 0] = pos['type']
            portfolio_obs[i, :, 1] = pos['size'] / self.initial_balance if self.initial_balance > 0 else 0
            portfolio_obs[i, :, 2] = pos['pnl']

        return np.concatenate([market_obs, portfolio_obs], axis=-1).astype(np.float32)

    def step(self, action):
        prev_portfolio_value = self.portfolio_value
        risk_penalty = self._execute_trade(action)

        self.index += 1
        self.portfolio_value = self._calculate_portfolio_value()
        self._check_stop_loss()

        pnl_reward = (self.portfolio_value / prev_portfolio_value) - 1
        scaled_pnl_reward = np.sign(pnl_reward) * np.log1p(abs(pnl_reward) * 100)
        holding_bonus = sum([1 for p in self.positions.values() if p['size'] > 0]) * self.holding_bonus_factor
        reward = scaled_pnl_reward + holding_bonus + risk_penalty

        terminated = self.index >= self.data_length - 2
        truncated = False
        if self.portfolio_value < self.initial_balance * 0.2:
            terminated = True
            reward = -1.0

        self._log_step_info()
        return self._get_obs(), reward, terminated, truncated, {}

    def _execute_trade(self, action):
        coin_idx, action_type, leverage_idx, size_idx = action
        if coin_idx >= self.num_coins: return 0

        coin_to_act = self.coin_names[coin_idx]
        current_price = self.prices[coin_idx, self.index, 3]
        risk_penalty = 0

        if action_type in [1, 2] and self.positions[coin_to_act]['size'] == 0:
            pos_type = 1 if action_type == 1 else -1
            leverage = self.leverage_tiers[leverage_idx]
            pos_size_ratio = self.position_size_tiers[size_idx]
            invest_amount = self.balance * pos_size_ratio

            if self.balance >= invest_amount > 0:
                self.balance -= invest_amount
                entry_price_slip = current_price * (1 + self.slippage_rate) if pos_type == 1 else current_price * (1 - self.slippage_rate)
                self.positions[coin_to_act] = {'type': pos_type, 'leverage': leverage, 'entry_price': entry_price_slip, 'size': invest_amount, 'pnl': 0}
                leverage_ratio = (leverage_idx + 1) / len(self.leverage_tiers)
                size_ratio = (size_idx + 1) / len(self.position_size_tiers)
                risk_penalty = -(leverage_ratio * size_ratio) * self.risk_penalty_factor

        elif action_type == 3 and self.positions[coin_to_act]['size'] > 0:
            p = self.positions[coin_to_act]
            exit_price_slip = current_price * (1 - self.slippage_rate) if p['type'] == 1 else current_price * (1 + self.slippage_rate)
            pnl = (exit_price_slip - p['entry_price']) * p['type'] * (p['size'] / p['entry_price']) * p['leverage']
            fee = p['size'] * p['leverage'] * self.trade_fee
            self.balance += (p['size'] + pnl - fee)
            self.positions[coin_to_act] = {'type': 0, 'leverage': 0, 'entry_price': 0, 'size': 0, 'pnl': 0}

        return risk_penalty

    def _calculate_portfolio_value(self):
        assets_value = 0
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        for name, p in self.positions.items():
            if p['size'] > 0:
                pnl = (current_prices[name] - p['entry_price']) * p['type'] * (p['size'] / p['entry_price']) * p['leverage']
                self.positions[name]['pnl'] = pnl
                assets_value += (p['size'] + pnl)
        return self.balance + assets_value

    def _check_stop_loss(self):
        current_prices = {name: self.prices[i, self.index, 3] for i, name in enumerate(self.coin_names)}
        for name, p in list(self.positions.items()):
            if p['size'] > 0:
                price_change_pct = (current_prices[name] - p['entry_price']) / p['entry_price'] * p['type']
                if price_change_pct < -self.stop_loss_pct:
                    self._execute_trade([self.coin_names.index(name), 3, 0, 0])

# AdvancedTradingEnv 클래스 내부
    def _log_step_info(self):
        """Logs the state of the portfolio at each step for backtesting."""
        date = self.pivot_df_index[self.index]
        holdings = {name: p['size'] * p['type'] * p['leverage'] for name, p in self.positions.items()}
        print(f"[LOG] {self.pivot_df_index[self.index]}: account_value={self.portfolio_value}")
        self.info_history.append({
            'date': date,
            'account_value': self.portfolio_value,
            'balance': self.balance,
            **holdings
        })

# --- 4. TRANSFORMER NETWORK ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class MultiAssetTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, d_model=128, nhead=8, num_layers=2, use_layernorm=False):
        num_coins, seq_len, num_features = observation_space.shape
        super().__init__(observation_space, features_dim=num_coins * d_model)
        self.d_model, self.num_coins = d_model, num_coins
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, activation='gelu', norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size * self.num_coins, x.shape[2], x.shape[3])
        projected = self.input_proj(x_reshaped)
        projected_with_pos = self.pos_encoder(projected)
        encoded = self.transformer(projected_with_pos)
        normalized_encoded = self.layer_norm(encoded)
        aggregated_features = normalized_encoded.mean(dim=1)
        output = aggregated_features.view(batch_size, self.num_coins * self.d_model)
        return output
# 이 테스트용 코드로 클래스 전체를 덮어쓰세요.
class SeparateActorCriticPolicy(ActorCriticPolicy):
    """
    A custom policy that uses two separate Transformer networks for the
    actor (policy) and critic (value) functions.
    """
    class MlpAlias:
        def __init__(self, policy):
            self.policy = policy
        def forward_actor(self, features):
            return self.policy.policy_extractor(features)
        def forward_critic(self, features):
            return self.policy.value_extractor(features)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        # --- Value Network (Critic) ---
        self.value_extractor = MultiAssetTransformerExtractor(
            self.observation_space, d_model=128, nhead=8, num_layers=4, use_layernorm=True
        )
        self.value_net = nn.Linear(self.value_extractor.features_dim, 1)

        # --- Policy Network (Actor) ---
        self.policy_extractor = MultiAssetTransformerExtractor(
            self.observation_space, d_model=64, nhead=4, num_layers=2
        )
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.policy_extractor.features_dim
        )
     
        # --- Optimizer ---
        all_params = list(self.value_extractor.parameters()) + list(self.value_net.parameters()) + \
                     list(self.policy_extractor.parameters()) + list(self.action_net.parameters())
        self.optimizer = self.optimizer_class(all_params, lr=lr_schedule(1), **self.optimizer_kwargs)
        self.mlp_extractor = SeparateActorCriticPolicy.MlpAlias(self)
    def forward(self, obs, deterministic=False):
        latent_vf = self.value_extractor(obs)
        values = self.value_net(latent_vf)
     
        latent_pi = self.policy_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
     
        return actions, values, log_prob
    def setup_model(self):
        # 이 메소드가 _build() 호출 뒤에 항상 실행됩니다.
        super().setup_model()
        # 외부에서 load 하더라도 mlp_extractor가 항상 보장되도록:
        if not hasattr(self, "mlp_extractor"):
            self.mlp_extractor = SeparateActorCriticPolicy.MlpAlias(self)

    def evaluate_actions(self, obs, actions):
        latent_vf = self.value_extractor(obs)
        values = self.value_net(latent_vf)
     
        latent_pi = self.policy_extractor(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
     
        return values, log_prob, entropy
     
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        latent_vf = self.value_extractor(obs)
        values = self.value_net(latent_vf)
        return values

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # vvvv 이 테스트용 프린트 문을 추가했습니다 vvvv
        print("\n>>> _predict 메소드가 성공적으로 호출되었습니다! <<<\n")
        
        latent_pi = self.policy_extractor(observation)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic=deterministic)
    def get_distribution(self, obs):
        """
        Get the policy distribution for the given observations.
        """
        latent_pi = self.policy_extractor(obs)
        return self._get_action_dist_from_latent(latent_pi)
def train_script(config: Config, start_stage=1):
    device = config.DEVICE
    print(f"Using device '{device}' for curriculum learning. Starting from stage: {start_stage}")
    model = None

    for stage_info in config.CURRICULUM_STAGES:
        stage = stage_info['stage']
        if stage < start_stage: continue

        print(f"\n===== STAGE {stage} BEGAN: {stage_info} =====\n")
        env_lambda = lambda: AdvancedTradingEnv(data_path=config.PARQUET_DATA_PATH, mode='parquet', window_size=config.WINDOW_SIZE, max_coins=stage_info['max_coins'], leverage_tiers=stage_info['leverage_tiers'], position_size_tiers=stage_info['position_size_tiers'], trade_fee=stage_info['trade_fee'], slippage_rate=stage_info['slippage_rate'])
        raw_env = DummyVecEnv([env_lambda])
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=config.PPO_PARAMS['gamma'])
        
        if model is None:
            if stage > 1:
                prev_model_path = f"{config.MODEL_PATH_PREFIX}_stage_{stage-1}.zip"
                if os.path.exists(prev_model_path):
                    print(f"Loading model from previous stage: {prev_model_path}")
                    model = PPO.load(prev_model_path, env=env, device=device, custom_objects={'learning_rate': config.PPO_PARAMS['learning_rate'], 'clip_range': config.PPO_PARAMS['clip_range']})
                else:
                    print(f"Stage {stage}: Creating a new model as no previous model was found.")
                    model = PPO(SeparateActorCriticPolicy, env, verbose=1, device=device, tensorboard_log=f"{config.LOG_DIR}/stage_{stage}", **config.PPO_PARAMS)
            else:
                 print(f"Stage {stage}: Creating a new model.")
                 model = PPO(SeparateActorCriticPolicy, env, verbose=1, device=device, tensorboard_log=f"{config.LOG_DIR}/stage_{stage}", **config.PPO_PARAMS)
        else:
            print(f"Stage {stage}: Transferring weights from Stage {stage-1} to new model.")
            new_model = PPO(SeparateActorCriticPolicy, env, verbose=1, device=device, tensorboard_log=f"{config.LOG_DIR}/stage_{stage}", **config.PPO_PARAMS)
            old_params, new_params = model.policy.state_dict(), new_model.policy.state_dict()
            print("--- Initiating Weight Transfer ---")
            for name, old_p in old_params.items():
                if name in new_params:
                    new_p = new_params[name]
                    if new_p.shape == old_p.shape:
                        new_p.data.copy_(old_p.data)
                    else:
                        print(f"  Shape mismatch for '{name}'. Attempting partial copy...")
                        if len(new_p.shape) >= 2 and len(old_p.shape) >= 2 and new_p.shape[1] > old_p.shape[1]:
                            new_p.data.zero_()
                            new_p.data[:, :old_p.shape[1]] = old_p.data
                            print(f"    -> Copied weights (Input: {old_p.shape[1]} -> {new_p.shape[1]})")
                        elif len(new_p.shape) == 1 and len(old_p.shape) == 1 and new_p.shape[0] > old_p.shape[0]:
                            new_p.data.zero_()
                            new_p.data[:old_p.shape[0]] = old_p.data
                            print(f"    -> Copied bias (Output: {old_p.shape[0]} -> {new_p.shape[0]})")
                        else:
                            print(f"    -> [WARNING] Unhandled shape mismatch for '{name}'. Skipping.")
            new_model.policy.load_state_dict(new_params)
            model = new_model
            print("--- Weight Transfer Complete ---")

        model.learn(total_timesteps=stage_info['total_timesteps'], reset_num_timesteps=False)
        model.save(f"{config.MODEL_PATH_PREFIX}_stage_{stage}.zip")
        env.save(f"{config.MODEL_PATH_PREFIX}_vecnorm_stage_{stage}.pkl")
        print(f"✅ STAGE {stage} model saved.")

    print("\n🎉 All curriculum learning stages complete!")
    if 'env' in locals() and env is not None: env.close()

# --- 6. BACKTESTING & EVALUATION SCRIPT ---
def get_daily_return(df, value_col_name="account_value"):
    df = df.copy()
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    return pd.Series(df["daily_return"].fillna(0), index=df.index)
def backtest_script(config: Config, stage_to_test: int):
    print(f"\n===== BACKTESTING STAGE {stage_to_test} MODEL =====\n")
    model_path = f"{config.MODEL_PATH_PREFIX}_stage_{stage_to_test}.zip"
    vecnorm_path = f"{config.MODEL_PATH_PREFIX}_vecnorm_stage_{stage_to_test}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
        print(f"Error: Cannot find model '{model_path}' or vecnorm '{vecnorm_path}'. Please train first.")
        return

    stage_info = config.CURRICULUM_STAGES[stage_to_test - 1]
    env_lambda = lambda: AdvancedTradingEnv(
        data_path=config.PARQUET_DATA_PATH,
        mode='parquet',
        window_size=config.WINDOW_SIZE,
        max_coins=stage_info['max_coins'],
        leverage_tiers=stage_info['leverage_tiers'],
        position_size_tiers=stage_info['position_size_tiers'],
        trade_fee=stage_info['trade_fee'],
        slippage_rate=stage_info['slippage_rate']
    )
    raw_env = DummyVecEnv([env_lambda])
    env = VecNormalize.load(vecnorm_path, raw_env)
    env.training, env.norm_reward = False, False

    model = PPO.load(
        model_path,
        env=env,
        device=config.DEVICE,
        custom_objects={"policy_class": SeparateActorCriticPolicy}
    )
    
    obs, terminated = env.reset(), False
    pbar = tqdm(total=env.get_attr('data_length')[0] - env.get_attr('window_size')[0])
    
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        terminated = dones[0]
        pbar.update(1)
    pbar.close()

    # FIX: Get the actual environment after simulation completes
    real_env = env.venv.envs[0]
    print(f"디버그: Final info_history 길이 = {len(real_env.info_history)}")
    
    # NEW: If still empty, try to access through env methods
    if len(real_env.info_history) == 0:
        print("Trying alternative access method...")
        # Force a final call to get_info_df
        results_df = real_env.get_info_df()
    else:
        results_df = pd.DataFrame(real_env.info_history)
    
    if results_df.empty:
        print("⚠️ 백테스트 로그가 여전히 비어 있습니다!")
        # Create a minimal CSV with account values from console output
        print("최소한의 결과를 저장합니다...")
        empty_df = pd.DataFrame({'message': ['백테스트가 실행되었지만 로그가 기록되지 않았습니다.']})
        empty_df.to_csv(f"backtest_results_stage_{stage_to_test}.csv", index=False)
        return

    print(f"백테스트 결과 컬럼: {results_df.columns.tolist()}")
    print(f"데이터 행 수: {len(results_df)}")
    print("첫 5행 미리보기:")
    print(results_df.head())

    results_df.to_csv(f"backtest_results_stage_{stage_to_test}.csv", index=False)
    print("✅ 백테스트 완료. 결과가 CSV에 저장되었습니다.")

    # QuantStats 리포트 생성
    print("QuantStats 성능 분석 리포트 생성 중...")
    try:
        returns = get_daily_return(results_df)
        if returns.index.tz is not None:
            returns.index = returns.index.tz_convert(None)
        
        report_filename = f'quantstats_report_stage_{stage_to_test}.html'
        quantstats.reports.html(
            returns,
            output=report_filename,
            title=f'Stage {stage_to_test} Performance'
        )
        print(f"✅ QuantStats 리포트가 '{report_filename}'에 저장되었습니다.")
    except Exception as e:
        print(f"⚠️ QuantStats 리포트 생성 중 오류: {e}")
        print("CSV 파일은 정상적으로 저장되었습니다.")
# --- 7. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", help="Execution mode: prepare_data, train, backtest", default="train")
    parser.add_argument("--stage", dest="stage", help="Stage to run (for training start or backtesting target)", type=int, default=1)
    options = parser.parse_args()
    config = Config()

    if options.mode == "prepare_data":
        prepare_data(config)
    elif options.mode == "train":
        if not os.path.exists(config.PARQUET_DATA_PATH):
             print(f"Data file '{config.PARQUET_DATA_PATH}' not found. Please run --mode=prepare_data first.")
        else:
             train_script(config, start_stage=options.stage)
    elif options.mode == "backtest":
        if not os.path.exists(config.PARQUET_DATA_PATH):
             print(f"Data file '{config.PARQUET_DATA_PATH}' not found. Please run --mode=prepare_data first.")
        else:
             backtest_script(config, stage_to_test=options.stage)
    else:
        raise ValueError("Invalid mode. Choose from 'prepare_data', 'train', 'backtest'.")
