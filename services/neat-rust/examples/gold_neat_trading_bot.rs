/**
 * NEAT Gold Trading Bot System
 * 
 * Dữ liệu lịch sử giá vàng lưu tại: gold_ohlcv_history.csv
 * Tạo quần thể NEAT 50 cá thể để giao dịch vàng.
 * Mỗi cá thể là 1 bot bắt đầu với $10,000 vốn.
 * Bot tự quyết định giao dịch (BUY, HOLD, SELL) nhằm tìm ra cách giao dịch tối ưu nhất.
 * 
 * Advanced Technical Indicators:
 * - 2MA30: MA của High 30 kỳ và MA của Low 30 kỳ
 * - MA Cross: MA 50 kỳ và MA 200 kỳ để xác định trend
 * - RSI, Bollinger Bands, Volatility
 * 
 * Lưu trữ neural network sau mỗi thế hệ để thống kê và tìm genome tốt nhất.
 */

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::{Deserialize, Serialize};
use neat_rust::{
    architecture::network::Network,
    utils::get_timestamp,
};

/// Dữ liệu OHLCV cho vàng một ngày
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldOhlcvData {
    pub date: String,
    pub timestamp: u64,
    pub open_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub close_price: f64,
    pub volume: f64,
}

/// Hành động giao dịch vàng
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoldTradingAction {
    Buy,
    Hold,
    Sell,
}

impl GoldTradingAction {
    /// Chuyển đổi từ output neural network thành action với threshold cho vàng
    pub fn from_output(output: f64) -> Self {
        if output < 0.35 {
            GoldTradingAction::Sell
        } else if output < 0.65 {
            GoldTradingAction::Hold
        } else {
            GoldTradingAction::Buy
        }
    }
}

/// Portfolio cho giao dịch vàng
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldPortfolio {
    pub initial_balance: f64,
    pub cash: f64,
    pub gold_holdings: f64,        // Số lượng vàng nắm giữ (ounces)
    pub total_value: f64,
    pub trades_count: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub max_drawdown: f64,
    pub peak_value: f64,
    pub trade_history: Vec<GoldTradeRecord>,
    pub consecutive_losses: u32,
}

impl GoldPortfolio {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            cash: initial_balance,
            gold_holdings: 0.0,
            total_value: initial_balance,
            trades_count: 0,
            winning_trades: 0,
            losing_trades: 0,
            max_drawdown: 0.0,
            peak_value: initial_balance,
            trade_history: Vec::new(),
            consecutive_losses: 0,
        }
    }

    /// Thực hiện giao dịch vàng với transaction cost
    pub fn execute_trade(&mut self, action: GoldTradingAction, price: f64, trade_percentage: f64) {
        let previous_value = self.total_value;
        let mut trade_executed = false;
        
        match action {
            GoldTradingAction::Buy => {
                let buy_amount = self.cash * trade_percentage;
                if buy_amount > 100.0 && self.cash >= buy_amount {
                    // Transaction cost cho vàng: 0.2% (cao hơn crypto)
                    let transaction_cost = buy_amount * 0.002;
                    let net_buy_amount = buy_amount - transaction_cost;
                    
                    let gold_bought = net_buy_amount / price;
                    self.cash -= buy_amount;
                    self.gold_holdings += gold_bought;
                    self.trades_count += 1;
                    trade_executed = true;
                    
                    self.trade_history.push(GoldTradeRecord {
                        action,
                        price,
                        amount: gold_bought,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                }
            }
            GoldTradingAction::Sell => {
                let sell_amount = self.gold_holdings * trade_percentage;
                if sell_amount > 0.001 && self.gold_holdings >= sell_amount {
                    let cash_received = sell_amount * price;
                    // Transaction cost cho vàng: 0.2%
                    let transaction_cost = cash_received * 0.002;
                    let net_cash_received = cash_received - transaction_cost;
                    
                    self.gold_holdings -= sell_amount;
                    self.cash += net_cash_received;
                    self.trades_count += 1;
                    trade_executed = true;
                    
                    self.trade_history.push(GoldTradeRecord {
                        action,
                        price,
                        amount: sell_amount,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                }
            }
            GoldTradingAction::Hold => {
                // Không làm gì
            }
        }
        
        // Cập nhật giá trị tổng
        self.update_value(price);
        
        // Cập nhật thống kê win/loss chỉ khi có giao dịch thực sự
        if trade_executed {
            if self.total_value > previous_value {
                self.winning_trades += 1;
                self.consecutive_losses = 0;
            } else if self.total_value < previous_value {
                self.losing_trades += 1;
                self.consecutive_losses += 1;
            }
        }
        
        // Cập nhật peak value và drawdown
        if self.total_value > self.peak_value {
            self.peak_value = self.total_value;
        } else {
            let drawdown = (self.peak_value - self.total_value) / self.peak_value;
            if drawdown > self.max_drawdown {
                self.max_drawdown = drawdown;
            }
        }
    }

    /// Cập nhật giá trị portfolio
    pub fn update_value(&mut self, current_price: f64) {
        self.total_value = self.cash + (self.gold_holdings * current_price);
    }

    /// Tính toán ROI
    pub fn roi(&self) -> f64 {
        (self.total_value - self.initial_balance) / self.initial_balance
    }

    /// Tính toán Sharpe ratio cho vàng
    pub fn sharpe_ratio(&self) -> f64 {
        let roi = self.roi();
        if self.max_drawdown > 0.0 {
            roi / self.max_drawdown
        } else {
            roi.max(0.0)
        }
    }

    /// Tính tỷ lệ thắng
    pub fn win_rate(&self) -> f64 {
        if self.trades_count > 0 {
            self.winning_trades as f64 / self.trades_count as f64
        } else {
            0.0
        }
    }
}

/// Bản ghi giao dịch vàng
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldTradeRecord {
    pub action: GoldTradingAction,
    pub price: f64,
    pub amount: f64,
    pub timestamp: u64,
}

/// Gold Trading Bot sử dụng NEAT
#[derive(Debug, Clone)]
pub struct GoldTradingBot {
    pub id: usize,
    pub network: Network,
    pub portfolio: GoldPortfolio,
    pub fitness: f64,
}

impl GoldTradingBot {
    pub fn new(id: usize, network: Network, initial_balance: f64) -> Self {
        Self {
            id,
            network,
            portfolio: GoldPortfolio::new(initial_balance),
            fitness: 0.0,
        }
    }

    /// Đưa ra quyết định giao dịch dựa trên dữ liệu thị trường vàng
    pub fn make_decision(&self, market_data: &[f64]) -> GoldTradingAction {
        let output = self.network.forward(market_data);
        GoldTradingAction::from_output(output[0])
    }

    /// Tính toán fitness score cho vàng với risk management
    pub fn calculate_fitness(&mut self) {
        let roi = self.portfolio.roi();
        let sharpe = self.portfolio.sharpe_ratio();
        let win_rate = self.portfolio.win_rate();
        
        // Fitness function tối ưu cho vàng
        let mut fitness = (roi * 0.35) + (sharpe * 0.25) + (win_rate * 0.25);
        
        // Penalty mạnh cho high drawdown (vàng cần bảo toàn vốn)
        if self.portfolio.max_drawdown > 0.3 {
            fitness *= 1.0 - (self.portfolio.max_drawdown * 0.8);
        }
        
        // Penalty cho over-trading
        if self.portfolio.trades_count > 500 {
            fitness *= 0.7;
        }
        
        // Penalty nghiêm cho under-trading
        if self.portfolio.trades_count < 20 {
            fitness *= 0.5;
        }
        
        // Bonus cho performance ổn định
        if roi > 0.1 && self.portfolio.max_drawdown < 0.15 && win_rate > 0.6 {
            fitness *= 1.3; // Bonus lớn cho performance vượt trội
        }
        
        // Penalty cho consecutive losses (quan trọng với vàng)
        if self.portfolio.consecutive_losses > 5 {
            fitness *= 0.8;
        }
        
        // Risk-adjusted return
        let risk_adjusted_return = roi / (1.0 + self.portfolio.max_drawdown);
        fitness += risk_adjusted_return * 0.15;
        
        self.fitness = fitness.max(0.0);
    }
}

/// Configuration cho indicators - cho phép bật/tắt từng chỉ báo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConfig {
    pub enable_ma_30: bool,           // Bật/tắt MA High/Low 30
    pub enable_ma_cross: bool,        // Bật/tắt MA Cross 50/200
    pub enable_rsi: bool,             // Bật/tắt RSI
    pub enable_bollinger_bands: bool, // Bật/tắt Bollinger Bands
    pub enable_volatility: bool,      // Bật/tắt Volatility
    pub enable_price_momentum: bool,  // Bật/tắt Price Momentum
    pub enable_volume_trend: bool,    // Bật/tắt Volume Trend
    pub enable_basic_ohlcv: bool,     // Bật/tắt OHLCV cơ bản
    pub lookback_days: usize,         // Số ngày lookback cho OHLCV
}

impl Default for IndicatorConfig {
    fn default() -> Self {
        Self {
            enable_ma_30: true,
            enable_ma_cross: true,
            enable_rsi: true,
            enable_bollinger_bands: true,
            enable_volatility: true,
            enable_price_momentum: true,
            enable_volume_trend: true,
            enable_basic_ohlcv: true,
            lookback_days: 5,
        }
    }
}

impl IndicatorConfig {
    /// Tạo config cho chiến lược cơ bản (chỉ OHLCV + MA)
    pub fn basic_strategy() -> Self {
        Self {
            enable_ma_30: true,
            enable_ma_cross: true,
            enable_rsi: false,
            enable_bollinger_bands: false,
            enable_volatility: false,
            enable_price_momentum: false,
            enable_volume_trend: false,
            enable_basic_ohlcv: true,
            lookback_days: 3,
        }
    }
    
    /// Tạo config cho chiến lược nâng cao (tất cả indicators)
    pub fn advanced_strategy() -> Self {
        Self::default()
    }
    
    /// Tạo config cho chiến lược momentum (tập trung vào momentum)
    pub fn momentum_strategy() -> Self {
        Self {
            enable_ma_30: false,
            enable_ma_cross: true,
            enable_rsi: true,
            enable_bollinger_bands: false,
            enable_volatility: true,
            enable_price_momentum: true,
            enable_volume_trend: true,
            enable_basic_ohlcv: true,
            lookback_days: 5,
        }
    }
    
    /// Tạo config cho chiến lược trend following (tập trung vào trend)
    pub fn trend_following_strategy() -> Self {
        Self {
            enable_ma_30: true,
            enable_ma_cross: true,
            enable_rsi: false,
            enable_bollinger_bands: true,
            enable_volatility: false,
            enable_price_momentum: true,
            enable_volume_trend: false,
            enable_basic_ohlcv: true,
            lookback_days: 10,
        }
    }
    
    /// Tạo config từ string (để dễ dàng thay đổi từ command line)
    pub fn from_strategy_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "basic" => Self::basic_strategy(),
            "advanced" => Self::advanced_strategy(),
            "momentum" => Self::momentum_strategy(),
            "trend" => Self::trend_following_strategy(),
            _ => {
                println!("Unknown strategy '{}', using default advanced strategy", name);
                Self::advanced_strategy()
            }
        }
    }
    
    /// Tạo custom config từ indicators list
    pub fn from_indicators(indicators: Vec<&str>, lookback_days: usize) -> Self {
        let mut config = Self {
            enable_ma_30: false,
            enable_ma_cross: false,
            enable_rsi: false,
            enable_bollinger_bands: false,
            enable_volatility: false,
            enable_price_momentum: false,
            enable_volume_trend: false,
            enable_basic_ohlcv: true, // Luôn bật OHLCV
            lookback_days,
        };
        
        for indicator in indicators {
            match indicator.to_lowercase().as_str() {
                "ma30" => config.enable_ma_30 = true,
                "ma_cross" => config.enable_ma_cross = true,
                "rsi" => config.enable_rsi = true,
                "bb" | "bollinger" => config.enable_bollinger_bands = true,
                "volatility" => config.enable_volatility = true,
                "momentum" => config.enable_price_momentum = true,
                "volume" => config.enable_volume_trend = true,
                "ohlcv" => config.enable_basic_ohlcv = true,
                _ => println!("Unknown indicator: {}", indicator),
            }
        }
        
        config
    }
    
    /// Load config từ JSON file
    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file_content = std::fs::read_to_string(file_path)?;
        let config: Self = serde_json::from_str(&file_content)?;
        Ok(config)
    }
    
    /// Save config to JSON file
    pub fn save_to_file(&self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(file_path, json)?;
        println!("Saved indicator config to: {}", file_path);
        Ok(())
    }
    
    /// Tính toán số lượng features sẽ được tạo ra
    pub fn calculate_feature_count(&self) -> usize {
        let mut count = 0;
        
        if self.enable_basic_ohlcv {
            count += self.lookback_days * 2; // close_price + volume cho mỗi ngày
        }
        
        if self.enable_ma_30 {
            count += 2; // ma_high_30 + ma_low_30
        }
        
        if self.enable_ma_cross {
            count += 3; // ma_50 + ma_200 + ma_cross_signal
        }
        
        if self.enable_rsi {
            count += 1; // rsi
        }
        
        if self.enable_bollinger_bands {
            count += 1; // bb_position
        }
        
        if self.enable_volatility {
            count += 1; // volatility
        }
        
        if self.enable_price_momentum {
            count += 1; // price_momentum
        }
        
        if self.enable_volume_trend {
            count += 1; // volume_trend
        }
        
        count
    }
    
    /// Lấy tên các indicators đang được bật
    pub fn get_active_indicators(&self) -> Vec<String> {
        let mut indicators = Vec::new();
        
        if self.enable_basic_ohlcv {
            indicators.push(format!("OHLCV({}d)", self.lookback_days));
        }
        if self.enable_ma_30 {
            indicators.push("MA30".to_string());
        }
        if self.enable_ma_cross {
            indicators.push("MA_Cross(50/200)".to_string());
        }
        if self.enable_rsi {
            indicators.push("RSI".to_string());
        }
        if self.enable_bollinger_bands {
            indicators.push("BB".to_string());
        }
        if self.enable_volatility {
            indicators.push("Volatility".to_string());
        }
        if self.enable_price_momentum {
            indicators.push("Momentum".to_string());
        }
        if self.enable_volume_trend {
            indicators.push("Volume_Trend".to_string());
        }
        
        indicators
    }
}

/// Advanced Technical Indicators cho vàng
#[derive(Debug, Clone)]
pub struct GoldTechnicalIndicators {
    pub ma_high_30: f64,      // MA của High 30 kỳ
    pub ma_low_30: f64,       // MA của Low 30 kỳ  
    pub ma_50: f64,           // MA 50 kỳ
    pub ma_200: f64,          // MA 200 kỳ
    pub ma_cross_signal: f64, // Signal từ MA Cross (1.0: bullish, -1.0: bearish, 0.0: neutral)
    pub rsi: f64,
    pub bb_position: f64,     // Vị trí trong Bollinger Bands
    pub volatility: f64,
    pub price_momentum: f64,
    pub volume_trend: f64,    // Trend của volume
}

/// Thống kê cho một thế hệ vàng
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldGenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub worst_fitness: f64,
    pub best_roi: f64,
    pub avg_roi: f64,
    pub best_win_rate: f64,
    pub avg_win_rate: f64,
    pub best_bot_id: usize,
    pub total_trades: u32,
    pub avg_trades: f64,
    pub best_portfolio: GoldPortfolio,
}

/// NEAT Gold Trading System với indicator configuration
pub struct NeatGoldTradingSystem {
    pub population_size: usize,
    pub initial_balance: f64,
    pub trade_percentage: f64,
    pub elite_percentage: f64,
    pub mutation_percentage: f64,
    pub current_generation: usize,
    pub market_data: Vec<GoldOhlcvData>,
    pub generation_stats: Vec<GoldGenerationStats>,
    pub indicator_config: IndicatorConfig,  // Thêm config cho indicators
}

impl NeatGoldTradingSystem {
    pub fn new(
        population_size: usize,
        initial_balance: f64,
        trade_percentage: f64,
        elite_percentage: f64,
        mutation_percentage: f64,
    ) -> Self {
        Self {
            population_size,
            initial_balance,
            trade_percentage,
            elite_percentage,
            mutation_percentage,
            current_generation: 0,
            market_data: Vec::new(),
            generation_stats: Vec::new(),
            indicator_config: IndicatorConfig::default(),
        }
    }
    
    /// Tạo với custom indicator config
    pub fn new_with_config(
        population_size: usize,
        initial_balance: f64,
        trade_percentage: f64,
        elite_percentage: f64,
        mutation_percentage: f64,
        indicator_config: IndicatorConfig,
    ) -> Self {
        Self {
            population_size,
            initial_balance,
            trade_percentage,
            elite_percentage,
            mutation_percentage,
            current_generation: 0,
            market_data: Vec::new(),
            generation_stats: Vec::new(),
            indicator_config,
        }
    }
    
    /// Thay đổi indicator config trong runtime
    pub fn set_indicator_config(&mut self, config: IndicatorConfig) {
        self.indicator_config = config;
        println!("Updated indicator config. Active indicators: {:?}", 
                 self.indicator_config.get_active_indicators());
    }
    
    /// Lấy thông tin về các indicators hiện đang sử dụng
    pub fn get_strategy_info(&self) -> String {
        format!(
            "Strategy: {} indicators active, {} features per sample",
            self.indicator_config.get_active_indicators().len(),
            self.indicator_config.calculate_feature_count()
        )
    }

    /// Load dữ liệu OHLCV vàng từ CSV
    pub fn load_gold_market_data(&mut self, csv_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(csv_path)?;
        let mut reader = csv::Reader::from_reader(BufReader::new(file));
        
        self.market_data.clear();
        
        for result in reader.deserialize() {
            let record: GoldOhlcvData = result?;
            self.market_data.push(record);
        }
        
        println!("Đã load {} ngày dữ liệu vàng từ {}", self.market_data.len(), csv_path);
        Ok(())
    }

    /// Tính toán Advanced Technical Indicators cho vàng
    pub fn calculate_gold_indicators(&self, day_index: usize, lookback: usize) -> GoldTechnicalIndicators {
        let start_idx = if day_index >= lookback { day_index - lookback + 1 } else { 0 };
        let end_idx = day_index + 1;
        
        if start_idx >= end_idx || end_idx > self.market_data.len() {
            return GoldTechnicalIndicators {
                ma_high_30: 0.0,
                ma_low_30: 0.0,
                ma_50: 0.0,
                ma_200: 0.0,
                ma_cross_signal: 0.0,
                rsi: 50.0,
                bb_position: 0.5,
                volatility: 0.0,
                price_momentum: 0.0,
                volume_trend: 0.0,
            };
        }
        
        let data_slice = &self.market_data[start_idx..end_idx];
        
        // MA High 30 và MA Low 30
        let (ma_high_30, ma_low_30) = if day_index >= 29 {
            let start_30 = day_index - 29;
            let high_30: Vec<f64> = self.market_data[start_30..=day_index]
                .iter().map(|d| d.high_price).collect();
            let low_30: Vec<f64> = self.market_data[start_30..=day_index]
                .iter().map(|d| d.low_price).collect();
            
            let ma_high = high_30.iter().sum::<f64>() / high_30.len() as f64;
            let ma_low = low_30.iter().sum::<f64>() / low_30.len() as f64;
            (ma_high, ma_low)
        } else {
            let highs: Vec<f64> = data_slice.iter().map(|d| d.high_price).collect();
            let lows: Vec<f64> = data_slice.iter().map(|d| d.low_price).collect();
            let ma_high = if !highs.is_empty() { highs.iter().sum::<f64>() / highs.len() as f64 } else { 0.0 };
            let ma_low = if !lows.is_empty() { lows.iter().sum::<f64>() / lows.len() as f64 } else { 0.0 };
            (ma_high, ma_low)
        };
        
        // MA Cross: MA 50 và MA 200
        let (ma_50, ma_200, ma_cross_signal) = if day_index >= 199 {
            let start_50 = day_index - 49;
            let start_200 = day_index - 199;
            
            let prices_50: Vec<f64> = self.market_data[start_50..=day_index]
                .iter().map(|d| d.close_price).collect();
            let prices_200: Vec<f64> = self.market_data[start_200..=day_index]
                .iter().map(|d| d.close_price).collect();
            
            let ma_50_val = prices_50.iter().sum::<f64>() / prices_50.len() as f64;
            let ma_200_val = prices_200.iter().sum::<f64>() / prices_200.len() as f64;
            
            // MA Cross Signal
            let cross_signal = if ma_50_val > ma_200_val {
                1.0 // Golden Cross - Bullish
            } else if ma_50_val < ma_200_val {
                -1.0 // Death Cross - Bearish  
            } else {
                0.0 // Neutral
            };
            
            (ma_50_val, ma_200_val, cross_signal)
        } else if day_index >= 49 {
            let start_50 = day_index - 49;
            let prices_50: Vec<f64> = self.market_data[start_50..=day_index]
                .iter().map(|d| d.close_price).collect();
            let ma_50_val = prices_50.iter().sum::<f64>() / prices_50.len() as f64;
            (ma_50_val, 0.0, 0.0)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        // RSI, Bollinger Bands, Volatility (như trước)
        let prices: Vec<f64> = data_slice.iter().map(|d| d.close_price).collect();
        let volumes: Vec<f64> = data_slice.iter().map(|d| d.volume).collect();
        
        // RSI
        let rsi = if prices.len() >= 2 {
            let mut gains = 0.0;
            let mut losses = 0.0;
            for i in 1..prices.len() {
                let change = prices[i] - prices[i-1];
                if change > 0.0 {
                    gains += change;
                } else {
                    losses += change.abs();
                }
            }
            let avg_gain = gains / (prices.len() - 1) as f64;
            let avg_loss = losses / (prices.len() - 1) as f64;
            if avg_loss > 0.0 {
                100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
            } else {
                100.0
            }
        } else {
            50.0
        };
        
        // Bollinger Bands
        let sma = if !prices.is_empty() { prices.iter().sum::<f64>() / prices.len() as f64 } else { 0.0 };
        let variance = if prices.len() > 1 {
            prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / prices.len() as f64
        } else {
            0.0
        };
        let volatility = variance.sqrt();
        
        let bb_position = if volatility > 0.0 {
            let upper_band = sma + (2.0 * volatility);
            let lower_band = sma - (2.0 * volatility);
            let current_price = prices.last().unwrap_or(&sma);
            (current_price - lower_band) / (upper_band - lower_band)
        } else {
            0.5
        };
        
        // Price momentum
        let price_momentum = if prices.len() >= 5 {
            let recent = prices.last().unwrap();
            let older = prices[prices.len() - 5];
            (recent - older) / older
        } else {
            0.0
        };
        
        // Volume trend
        let volume_trend = if volumes.len() >= 5 {
            let recent_vol = volumes.iter().rev().take(3).sum::<f64>() / 3.0;
            let older_vol = volumes.iter().rev().skip(3).take(3).sum::<f64>() / 3.0;
            if older_vol > 0.0 {
                (recent_vol - older_vol) / older_vol
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        GoldTechnicalIndicators {
            ma_high_30,
            ma_low_30,
            ma_50,
            ma_200,
            ma_cross_signal,
            rsi: rsi / 100.0, // Normalize
            bb_position,
            volatility: volatility / 1000.0, // Normalize cho vàng
            price_momentum,
            volume_trend,
        }
    }

    /// Chuẩn bị dữ liệu đầu vào cho neural network với dynamic indicators
    pub fn prepare_gold_market_features(&self, day_index: usize) -> Vec<f64> {
        let mut features = Vec::new();
        
        // Basic OHLCV features (nếu được bật)
        if self.indicator_config.enable_basic_ohlcv {
            let lookback = self.indicator_config.lookback_days;
            
            if day_index >= lookback && day_index < self.market_data.len() {
                for i in (day_index - lookback + 1)..=day_index {
                    let data = &self.market_data[i];
                    features.extend_from_slice(&[
                        data.close_price / 1000.0,    // Normalize giá vàng
                        data.volume / 100000.0,       // Normalize volume vàng
                    ]);
                }
            } else {
                // Fallback với dữ liệu hiện tại
                let current = &self.market_data[day_index.min(self.market_data.len() - 1)];
                for _ in 0..lookback {
                    features.extend_from_slice(&[
                        current.close_price / 1000.0,
                        current.volume / 100000.0,
                    ]);
                }
            }
        }
        
        // Advanced Technical Indicators (chỉ tính nếu có indicator nào được bật)
        if self.indicator_config.enable_ma_30 || 
           self.indicator_config.enable_ma_cross || 
           self.indicator_config.enable_rsi ||
           self.indicator_config.enable_bollinger_bands ||
           self.indicator_config.enable_volatility ||
           self.indicator_config.enable_price_momentum ||
           self.indicator_config.enable_volume_trend {
            
            let indicators = self.calculate_gold_indicators(day_index, 
                                                           self.indicator_config.lookback_days);
            
            // Thêm các indicators đã được bật
            if self.indicator_config.enable_ma_30 {
                features.extend_from_slice(&[
                    indicators.ma_high_30 / 1000.0,
                    indicators.ma_low_30 / 1000.0,
                ]);
            }
            
            if self.indicator_config.enable_ma_cross {
                features.extend_from_slice(&[
                    indicators.ma_50 / 1000.0,
                    indicators.ma_200 / 1000.0,
                    indicators.ma_cross_signal,     // Đã chuẩn hóa -1 to 1
                ]);
            }
            
            if self.indicator_config.enable_rsi {
                features.push(indicators.rsi);      // Đã chuẩn hóa 0 to 1
            }
            
            if self.indicator_config.enable_bollinger_bands {
                features.push(indicators.bb_position);
            }
            
            if self.indicator_config.enable_volatility {
                features.push(indicators.volatility);
            }
            
            if self.indicator_config.enable_price_momentum {
                features.push(indicators.price_momentum);
            }
            
            if self.indicator_config.enable_volume_trend {
                features.push(indicators.volume_trend);
            }
        }
        
        // Đảm bảo luôn có ít nhất 1 feature
        if features.is_empty() {
            let current = &self.market_data[day_index.min(self.market_data.len() - 1)];
            features.push(current.close_price / 1000.0);
        }
        
        features
    }

    /// Tạo quần thể ban đầu với dynamic feature count
    pub fn create_initial_population(&self) -> Vec<GoldTradingBot> {
        let mut population = Vec::new();
        let feature_count = self.indicator_config.calculate_feature_count();
        
        println!("Creating population with {} input features", feature_count);
        
        for i in 0..self.population_size {
            let network = Network::new(feature_count, 1);
            let bot = GoldTradingBot::new(i, network, self.initial_balance);
            population.push(bot);
        }
        
        population
    }

    /// Chạy simulation cho một thế hệ
    pub fn run_simulation(&self, population: &mut Vec<GoldTradingBot>) {
        for bot in population.iter_mut() {
            // Reset portfolio
            bot.portfolio = GoldPortfolio::new(self.initial_balance);
            
            // Chạy qua tất cả dữ liệu thị trường vàng
            for (day_index, market_day) in self.market_data.iter().enumerate() {
                if day_index == 0 {
                    continue; // Bỏ qua ngày đầu tiên
                }
                
                // Chuẩn bị features với dynamic indicators
                let features = self.prepare_gold_market_features(day_index);
                
                // Bot đưa ra quyết định
                let action = bot.make_decision(&features);
                
                // Thực hiện giao dịch
                bot.portfolio.execute_trade(action, market_day.close_price, self.trade_percentage);
                
                // Cập nhật giá trị portfolio
                bot.portfolio.update_value(market_day.close_price);
            }
            
            // Tính toán fitness
            bot.calculate_fitness();
        }
    }

    /// Tạo thế hệ mới
    pub fn create_next_generation(&self, current_population: &[GoldTradingBot]) -> Vec<GoldTradingBot> {
        let mut sorted_population = current_population.to_vec();
        sorted_population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        let elite_count = ((self.population_size as f64) * self.elite_percentage).round() as usize;
        let mutation_count = ((self.population_size as f64) * self.mutation_percentage).round() as usize;
        
        let mut next_generation = Vec::new();
        
        // Giữ lại elite
        for i in 0..elite_count {
            let mut bot = sorted_population[i].clone();
            bot.id = next_generation.len();
            bot.portfolio = GoldPortfolio::new(self.initial_balance);
            bot.fitness = 0.0;
            next_generation.push(bot);
        }
        
        // Tạo mutation từ elite
        for _ in 0..mutation_count {
            let parent_idx = rand::random::<usize>() % elite_count;
            let mut mutated_network = sorted_population[parent_idx].network.clone();
            mutated_network.mutate();
            
            let mut bot = GoldTradingBot::new(
                next_generation.len(),
                mutated_network,
                self.initial_balance,
            );
            bot.fitness = 0.0;
            next_generation.push(bot);
        }
        
        // Tạo crossover để fill phần còn lại
        while next_generation.len() < self.population_size {
            let parent1_idx = rand::random::<usize>() % elite_count;
            let parent2_idx = rand::random::<usize>() % elite_count;
            
            if parent1_idx != parent2_idx {
                // Crossover: trộn weights
                let mut child_network = sorted_population[parent1_idx].network.clone();
                let parent2_network = &sorted_population[parent2_idx].network;
                
                // Trộn 50% weights từ parent2
                for (i, weight) in child_network.weights.iter_mut().enumerate() {
                    if rand::random::<bool>() && i < parent2_network.weights.len() {
                        *weight = parent2_network.weights[i];
                    }
                }
                
                let mut bot = GoldTradingBot::new(
                    next_generation.len(),
                    child_network,
                    self.initial_balance,
                );
                bot.fitness = 0.0;
                next_generation.push(bot);
            }
        }
        
        next_generation
    }

    /// Tính toán thống kê cho thế hệ
    pub fn calculate_generation_stats(&self, population: &[GoldTradingBot]) -> GoldGenerationStats {
        let mut fitness_values: Vec<f64> = population.iter().map(|bot| bot.fitness).collect();
        let mut roi_values: Vec<f64> = population.iter().map(|bot| bot.portfolio.roi()).collect();
        let mut win_rates: Vec<f64> = population.iter().map(|bot| bot.portfolio.win_rate()).collect();
        
        fitness_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        roi_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        win_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let best_bot = population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
        let total_trades: u32 = population.iter().map(|bot| bot.portfolio.trades_count).sum();
        
        GoldGenerationStats {
            generation: self.current_generation,
            best_fitness: fitness_values.last().copied().unwrap_or(0.0),
            avg_fitness: fitness_values.iter().sum::<f64>() / fitness_values.len() as f64,
            worst_fitness: fitness_values.first().copied().unwrap_or(0.0),
            best_roi: roi_values.last().copied().unwrap_or(0.0),
            avg_roi: roi_values.iter().sum::<f64>() / roi_values.len() as f64,
            best_win_rate: win_rates.last().copied().unwrap_or(0.0),
            avg_win_rate: win_rates.iter().sum::<f64>() / win_rates.len() as f64,
            best_bot_id: best_bot.id,
            total_trades,
            avg_trades: total_trades as f64 / population.len() as f64,
            best_portfolio: best_bot.portfolio.clone(),
        }
    }

    /// Chạy evolution cho nhiều thế hệ
    pub fn run_evolution(&mut self, generations: usize) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== NEAT GOLD TRADING BOT EVOLUTION ===");
        println!("Population: {}", self.population_size);
        println!("Initial Balance: ${}", self.initial_balance);
        println!("Trade Percentage: {}%", self.trade_percentage * 100.0);
        println!("Elite: {}%, Mutation: {}%", self.elite_percentage * 100.0, self.mutation_percentage * 100.0);
        println!("Gold Market Data: {} days", self.market_data.len());
        println!("Strategy: {}", self.get_strategy_info());
        println!("Active Indicators: {:?}", self.indicator_config.get_active_indicators());
        println!();

        let mut population = self.create_initial_population();
        
        for generation in 0..generations {
            self.current_generation = generation;
            
            println!("Generation {}/{}", generation + 1, generations);
            
            // Chạy simulation
            self.run_simulation(&mut population);
            
            // Tính toán thống kê
            let stats = self.calculate_generation_stats(&population);
            
            // In thống kê
            println!("  Best Fitness: {:.4}", stats.best_fitness);
            println!("  Best ROI: {:.2}%", stats.best_roi * 100.0);
            println!("  Avg ROI: {:.2}%", stats.avg_roi * 100.0);
            println!("  Best Win Rate: {:.1}%", stats.best_win_rate * 100.0);
            println!("  Best Bot: #{} (${:.2})", stats.best_bot_id, stats.best_portfolio.total_value);
            println!("  Avg Trades: {:.1}", stats.avg_trades);
            println!("  Max Drawdown: {:.2}%", stats.best_portfolio.max_drawdown * 100.0);
            println!("  Sharpe Ratio: {:.3}", stats.best_portfolio.sharpe_ratio());
            
            // Lưu thống kê
            self.generation_stats.push(stats);
            
            // Lưu best bot của thế hệ này
            let best_bot = population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
            let timestamp = get_timestamp();
            let bot_path = format!("models/gold_trading_bot_gen{}_{}.json", generation, timestamp);
            if let Err(e) = best_bot.network.save_to_json(Path::new(&bot_path)) {
                println!("  Warning: Không thể lưu bot: {}", e);
            } else {
                println!("  Saved best gold bot: {}", bot_path);
            }
            
            // Tạo thế hệ mới (trừ thế hệ cuối)
            if generation < generations - 1 {
                population = self.create_next_generation(&population);
            }
            
            println!();
        }
        
        // Lưu thống kê cuối cùng
        self.save_final_results()?;
        
        Ok(())
    }

    /// Lưu kết quả cuối cùng
    pub fn save_final_results(&self) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = get_timestamp();
        let stats_path = format!("models/gold_evolution_stats_{}.json", timestamp);
        
        let json = serde_json::to_string_pretty(&self.generation_stats)?;
        std::fs::write(&stats_path, json)?;
        
        println!("=== GOLD EVOLUTION COMPLETE ===");
        println!("Saved evolution stats: {}", stats_path);
        
        // Tìm thế hệ tốt nhất
        if let Some(best_gen) = self.generation_stats.iter().max_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap()) {
            println!("\nBEST GOLD GENERATION: {}", best_gen.generation);
            println!("  Best Fitness: {:.4}", best_gen.best_fitness);
            println!("  Best ROI: {:.2}%", best_gen.best_roi * 100.0);
            println!("  Best Win Rate: {:.1}%", best_gen.best_win_rate * 100.0);
            println!("  Final Value: ${:.2}", best_gen.best_portfolio.total_value);
            println!("  Total Trades: {}", best_gen.best_portfolio.trades_count);
            println!("  Max Drawdown: {:.2}%", best_gen.best_portfolio.max_drawdown * 100.0);
            println!("  Sharpe Ratio: {:.3}", best_gen.best_portfolio.sharpe_ratio());
        }
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NEAT Gold Trading Bot System with Dynamic Indicators");
    println!("====================================================");
    
    // Demo các strategy config khác nhau
    println!("\n=== AVAILABLE STRATEGIES ===");
    let basic_config = IndicatorConfig::basic_strategy();
    let advanced_config = IndicatorConfig::advanced_strategy();
    let momentum_config = IndicatorConfig::momentum_strategy();
    let trend_config = IndicatorConfig::trend_following_strategy();
    
    println!("1. Basic Strategy: {} features, indicators: {:?}", 
             basic_config.calculate_feature_count(), basic_config.get_active_indicators());
    println!("2. Advanced Strategy: {} features, indicators: {:?}", 
             advanced_config.calculate_feature_count(), advanced_config.get_active_indicators());
    println!("3. Momentum Strategy: {} features, indicators: {:?}", 
             momentum_config.calculate_feature_count(), momentum_config.get_active_indicators());
    println!("4. Trend Following Strategy: {} features, indicators: {:?}", 
             trend_config.calculate_feature_count(), trend_config.get_active_indicators());
    
    // Chọn strategy để chạy - có thể thay đổi strategy ở đây
    let selected_strategy = advanced_config; // Thay đổi strategy tại đây
    
    println!("\n=== SELECTED STRATEGY ===");
    println!("Running with: {} features", selected_strategy.calculate_feature_count());
    println!("Active indicators: {:?}", selected_strategy.get_active_indicators());
    
    // Khởi tạo hệ thống trading vàng với strategy đã chọn
    let mut gold_trading_system = NeatGoldTradingSystem::new_with_config(
        50,      // 50 bots trong quần thể
        10000.0, // $10,000 ban đầu
        0.02,    // 2% mỗi lần giao dịch
        0.10,    // Giữ lại 10% elite
        0.20,    // 20% mutation
        selected_strategy,
    );
    
    // Load dữ liệu vàng
    gold_trading_system.load_gold_market_data("examples/gold_ohlcv_history.csv")?;
    
    // Chạy evolution cho 50 thế hệ
    gold_trading_system.run_evolution(50)?;
    
    println!("\n=== STRATEGY COMPARISON DEMO ===");
    println!("You can easily switch strategies by modifying the 'selected_strategy' variable in main()");
    println!("Each strategy will create different neural network architectures automatically!");
    
    println!("\nGold Evolution completed successfully!");
    println!("Check models/ folder for saved neural networks and statistics.");
    
    Ok(())
}
