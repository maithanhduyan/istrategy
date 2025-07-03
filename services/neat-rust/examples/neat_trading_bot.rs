use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::{Deserialize, Serialize};
use neat_rust::{
    architecture::network::Network,
    utils::get_timestamp,
};

/// Dữ liệu OHLCV cho một ngày
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvData {
    pub date: String,
    pub timestamp: u64,
    pub open_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub close_price: f64,
    pub volume: f64,
}

/// Hành động giao dịch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingAction {
    Buy,
    Hold,
    Sell,
}

impl TradingAction {
    /// Chuyển đổi từ output neural network thành action
    pub fn from_output(output: f64) -> Self {
        if output < 0.33 {
            TradingAction::Sell
        } else if output < 0.67 {
            TradingAction::Hold
        } else {
            TradingAction::Buy
        }
    }
}

/// Portfolio của một trading bot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub initial_balance: f64,
    pub cash: f64,
    pub btc_holdings: f64,
    pub total_value: f64,
    pub trades_count: u32,
    pub winning_trades: u32,
    pub losing_trades: u32,
    pub max_drawdown: f64,
    pub peak_value: f64,
    pub trade_history: Vec<TradeRecord>,
}

impl Portfolio {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            initial_balance,
            cash: initial_balance,
            btc_holdings: 0.0,
            total_value: initial_balance,
            trades_count: 0,
            winning_trades: 0,
            losing_trades: 0,
            max_drawdown: 0.0,
            peak_value: initial_balance,
            trade_history: Vec::new(),
        }
    }

    /// Thực hiện giao dịch
    pub fn execute_trade(&mut self, action: TradingAction, price: f64, trade_percentage: f64) {
        let previous_value = self.total_value;
        let mut trade_executed = false;
        
        match action {
            TradingAction::Buy => {
                let buy_amount = self.cash * trade_percentage;
                if buy_amount > 0.0 && self.cash >= buy_amount {
                    // Thêm transaction cost (0.1%)
                    let transaction_cost = buy_amount * 0.001;
                    let net_buy_amount = buy_amount - transaction_cost;
                    
                    let btc_bought = net_buy_amount / price;
                    self.cash -= buy_amount; // Trừ cả phí giao dịch
                    self.btc_holdings += btc_bought;
                    self.trades_count += 1;
                    trade_executed = true;
                    
                    self.trade_history.push(TradeRecord {
                        action,
                        price,
                        amount: btc_bought,
                        timestamp: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    });
                }
            }
            TradingAction::Sell => {
                let sell_amount = self.btc_holdings * trade_percentage;
                if sell_amount > 0.0 && self.btc_holdings >= sell_amount {
                    let cash_received = sell_amount * price;
                    // Thêm transaction cost (0.1%)
                    let transaction_cost = cash_received * 0.001;
                    let net_cash_received = cash_received - transaction_cost;
                    
                    self.btc_holdings -= sell_amount;
                    self.cash += net_cash_received;
                    self.trades_count += 1;
                    trade_executed = true;
                    
                    self.trade_history.push(TradeRecord {
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
            TradingAction::Hold => {
                // Không làm gì
            }
        }
        
        // Cập nhật giá trị tổng
        self.update_value(price);
        
        // Cập nhật thống kê win/loss chỉ khi có giao dịch thực sự
        if trade_executed {
            if self.total_value > previous_value {
                self.winning_trades += 1;
            } else if self.total_value < previous_value {
                self.losing_trades += 1;
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
        self.total_value = self.cash + (self.btc_holdings * current_price);
    }

    /// Tính toán ROI
    pub fn roi(&self) -> f64 {
        (self.total_value - self.initial_balance) / self.initial_balance
    }

    /// Tính toán Sharpe ratio (đơn giản)
    pub fn sharpe_ratio(&self) -> f64 {
        let roi = self.roi();
        if self.max_drawdown > 0.0 {
            roi / self.max_drawdown
        } else {
            roi
        }
    }
}

/// Bản ghi giao dịch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub action: TradingAction,
    pub price: f64,
    pub amount: f64,
    pub timestamp: u64,
}

/// Trading Bot sử dụng NEAT
#[derive(Debug, Clone)]
pub struct TradingBot {
    pub id: usize,
    pub network: Network,
    pub portfolio: Portfolio,
    pub fitness: f64,
}

impl TradingBot {
    pub fn new(id: usize, network: Network, initial_balance: f64) -> Self {
        Self {
            id,
            network,
            portfolio: Portfolio::new(initial_balance),
            fitness: 0.0,
        }
    }

    /// Đưa ra quyết định giao dịch dựa trên dữ liệu thị trường
    pub fn make_decision(&self, market_data: &[f64]) -> TradingAction {
        let output = self.network.forward(market_data);
        TradingAction::from_output(output[0])
    }

    /// Tính toán fitness score
    pub fn calculate_fitness(&mut self) {
        let roi = self.portfolio.roi();
        let sharpe = self.portfolio.sharpe_ratio();
        let trade_efficiency = if self.portfolio.trades_count > 0 {
            self.portfolio.winning_trades as f64 / self.portfolio.trades_count as f64
        } else {
            0.0
        };

        // Công thức fitness cải tiến: kết hợp ROI, Sharpe ratio, trade efficiency và penalty cho high drawdown
        let mut fitness = (roi * 0.4) + (sharpe * 0.3) + (trade_efficiency * 0.2);
        
        // Thêm penalty cho high drawdown
        if self.portfolio.max_drawdown > 0.5 {
            fitness *= 1.0 - self.portfolio.max_drawdown * 0.5; // Giảm fitness nếu drawdown > 50%
        }
        
        // Penalty cho quá ít giao dịch hoặc quá nhiều giao dịch
        if self.portfolio.trades_count < 10 {
            fitness *= 0.7; // Penalty cho ít giao dịch
        } else if self.portfolio.trades_count > 2000 {
            fitness *= 0.8; // Penalty cho quá nhiều giao dịch
        }
        
        // Bonus cho consistent performance
        if roi > 0.0 && self.portfolio.max_drawdown < 0.3 {
            fitness *= 1.1; // Bonus cho low drawdown với positive ROI
        }
        
        self.fitness = fitness;
    }
}

/// Thống kê cho một thế hệ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub worst_fitness: f64,
    pub best_roi: f64,
    pub avg_roi: f64,
    pub best_bot_id: usize,
    pub total_trades: u32,
    pub avg_trades: f64,
    pub best_portfolio: Portfolio,
}

/// NEAT Trading System
pub struct NeatTradingSystem {
    pub population_size: usize,
    pub initial_balance: f64,
    pub trade_percentage: f64,
    pub elite_percentage: f64,
    pub mutation_percentage: f64,
    pub current_generation: usize,
    pub market_data: Vec<OhlcvData>,
    pub generation_stats: Vec<GenerationStats>,
}

impl NeatTradingSystem {
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
        }
    }

    /// Load dữ liệu OHLCV từ CSV
    pub fn load_market_data(&mut self, csv_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(csv_path)?;
        let mut reader = csv::Reader::from_reader(BufReader::new(file));
        
        self.market_data.clear();
        
        for result in reader.deserialize() {
            let record: OhlcvData = result?;
            self.market_data.push(record);
        }
        
        println!("Đã load {} ngày dữ liệu BTC từ {}", self.market_data.len(), csv_path);
        Ok(())
    }

    /// Chuẩn bị dữ liệu đầu vào cho neural network
    pub fn prepare_market_features(&self, day_index: usize, lookback: usize) -> Vec<f64> {
        let mut features = Vec::new();
        
        if day_index < lookback {
            // Không đủ dữ liệu lịch sử, sử dụng dữ liệu hiện tại
            let current = &self.market_data[day_index];
            features.extend_from_slice(&[
                current.open_price / 10000.0,  // Normalize
                current.high_price / 10000.0,
                current.low_price / 10000.0,
                current.close_price / 10000.0,
                current.volume / 1e8,  // Normalize volume
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // Padding for missing indicators
            ]);
        } else {
            // Sử dụng dữ liệu của lookback ngày gần nhất
            for i in (day_index - lookback + 1)..=day_index {
                let data = &self.market_data[i];
                features.extend_from_slice(&[
                    data.close_price / 10000.0,
                    data.volume / 1e8,
                ]);
            }
            
            // Thêm technical indicators nâng cao
            let prices: Vec<f64> = self.market_data[(day_index - lookback + 1)..=day_index]
                .iter()
                .map(|d| d.close_price)
                .collect();
            
            let _highs: Vec<f64> = self.market_data[(day_index - lookback + 1)..=day_index]
                .iter()
                .map(|d| d.high_price)
                .collect();
            
            let _lows: Vec<f64> = self.market_data[(day_index - lookback + 1)..=day_index]
                .iter()
                .map(|d| d.low_price)
                .collect();
            
            // Simple Moving Average
            let sma = prices.iter().sum::<f64>() / prices.len() as f64;
            features.push(sma / 10000.0);
            
            // Price change percentage
            let price_change = (prices.last().unwrap() - prices.first().unwrap()) / prices.first().unwrap();
            features.push(price_change);
            
            // Volatility (standard deviation)
            let mean = sma;
            let variance = prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / prices.len() as f64;
            let volatility = variance.sqrt();
            features.push(volatility / 10000.0);
            
            // RSI (Relative Strength Index) - simplified
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
            let rsi = if avg_loss > 0.0 {
                100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
            } else {
                100.0
            };
            features.push(rsi / 100.0); // Normalize RSI
            
            // Bollinger Bands
            let upper_band = sma + (2.0 * volatility);
            let lower_band = sma - (2.0 * volatility);
            let current_price = prices.last().unwrap();
            let bb_position = (current_price - lower_band) / (upper_band - lower_band);
            features.push(bb_position);
            
            // Price momentum (rate of change)
            let momentum = if prices.len() >= 3 {
                (prices.last().unwrap() - prices[prices.len() - 3]) / prices[prices.len() - 3]
            } else {
                0.0
            };
            features.push(momentum);
        }
        
        features
    }

    /// Tạo quần thể ban đầu
    pub fn create_initial_population(&self) -> Vec<TradingBot> {
        let mut population = Vec::new();
        
        for i in 0..self.population_size {
            let network = Network::new(21, 1); // 21 input features, 1 output (action)
            let bot = TradingBot::new(i, network, self.initial_balance);
            population.push(bot);
        }
        
        population
    }

    /// Chạy simulation cho một thế hệ
    pub fn run_simulation(&self, population: &mut Vec<TradingBot>) {
        let lookback = 5; // Sử dụng 5 ngày dữ liệu trước đó
        
        for bot in population.iter_mut() {
            // Reset portfolio
            bot.portfolio = Portfolio::new(self.initial_balance);
            
            // Chạy qua tất cả dữ liệu thị trường
            for (day_index, market_day) in self.market_data.iter().enumerate() {
                if day_index == 0 {
                    continue; // Bỏ qua ngày đầu tiên
                }
                
                // Chuẩn bị features
                let features = self.prepare_market_features(day_index, lookback);
                
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
    pub fn create_next_generation(&self, current_population: &[TradingBot]) -> Vec<TradingBot> {
        let mut sorted_population = current_population.to_vec();
        sorted_population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        
        let elite_count = ((self.population_size as f64) * self.elite_percentage).round() as usize;
        let mutation_count = ((self.population_size as f64) * self.mutation_percentage).round() as usize;
        
        let mut next_generation = Vec::new();
        
        // Giữ lại elite
        for i in 0..elite_count {
            let mut bot = sorted_population[i].clone();
            bot.id = next_generation.len();
            bot.portfolio = Portfolio::new(self.initial_balance);
            bot.fitness = 0.0;
            next_generation.push(bot);
        }
        
        // Tạo mutation từ elite
        for _ in 0..mutation_count {
            let parent_idx = rand::random::<usize>() % elite_count;
            let mut mutated_network = sorted_population[parent_idx].network.clone();
            mutated_network.mutate();
            
            let mut bot = TradingBot::new(
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
                // Crossover đơn giản: trộn weights
                let mut child_network = sorted_population[parent1_idx].network.clone();
                let parent2_network = &sorted_population[parent2_idx].network;
                
                // Trộn 50% weights từ parent2
                for (i, weight) in child_network.weights.iter_mut().enumerate() {
                    if rand::random::<bool>() && i < parent2_network.weights.len() {
                        *weight = parent2_network.weights[i];
                    }
                }
                
                let mut bot = TradingBot::new(
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
    pub fn calculate_generation_stats(&self, population: &[TradingBot]) -> GenerationStats {
        let mut fitness_values: Vec<f64> = population.iter().map(|bot| bot.fitness).collect();
        let mut roi_values: Vec<f64> = population.iter().map(|bot| bot.portfolio.roi()).collect();
        
        fitness_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        roi_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let best_bot = population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
        let total_trades: u32 = population.iter().map(|bot| bot.portfolio.trades_count).sum();
        
        GenerationStats {
            generation: self.current_generation,
            best_fitness: fitness_values.last().copied().unwrap_or(0.0),
            avg_fitness: fitness_values.iter().sum::<f64>() / fitness_values.len() as f64,
            worst_fitness: fitness_values.first().copied().unwrap_or(0.0),
            best_roi: roi_values.last().copied().unwrap_or(0.0),
            avg_roi: roi_values.iter().sum::<f64>() / roi_values.len() as f64,
            best_bot_id: best_bot.id,
            total_trades,
            avg_trades: total_trades as f64 / population.len() as f64,
            best_portfolio: best_bot.portfolio.clone(),
        }
    }

    /// Chạy evolution cho nhiều thế hệ
    pub fn run_evolution(&mut self, generations: usize) -> Result<(), Box<dyn std::error::Error>> {
        println!("=== NEAT TRADING BOT EVOLUTION ===");
        println!("Population: {}", self.population_size);
        println!("Initial Balance: ${}", self.initial_balance);
        println!("Trade Percentage: {}%", self.trade_percentage * 100.0);
        println!("Elite: {}%, Mutation: {}%", self.elite_percentage * 100.0, self.mutation_percentage * 100.0);
        println!("Market Data: {} days", self.market_data.len());
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
            println!("  Best Bot: #{} (${:.2})", stats.best_bot_id, stats.best_portfolio.total_value);
            println!("  Avg Trades: {:.1}", stats.avg_trades);
            println!("  Max Drawdown: {:.2}%", stats.best_portfolio.max_drawdown * 100.0);
            
            // Lưu thống kê
            self.generation_stats.push(stats);
            
            // Lưu best bot của thế hệ này
            let best_bot = population.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()).unwrap();
            let timestamp = get_timestamp();
            let bot_path = format!("models/trading_bot_gen{}_{}.json", generation, timestamp);
            if let Err(e) = best_bot.network.save_to_json(Path::new(&bot_path)) {
                println!("  Warning: Không thể lưu bot: {}", e);
            } else {
                println!("  Saved best bot: {}", bot_path);
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
        let stats_path = format!("models/trading_evolution_stats_{}.json", timestamp);
        
        let json = serde_json::to_string_pretty(&self.generation_stats)?;
        std::fs::write(&stats_path, json)?;
        
        println!("=== EVOLUTION COMPLETE ===");
        println!("Saved evolution stats: {}", stats_path);
        
        // Tìm thế hệ tốt nhất
        if let Some(best_gen) = self.generation_stats.iter().max_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap()) {
            println!("\nBEST GENERATION: {}", best_gen.generation);
            println!("  Best Fitness: {:.4}", best_gen.best_fitness);
            println!("  Best ROI: {:.2}%", best_gen.best_roi * 100.0);
            println!("  Final Value: ${:.2}", best_gen.best_portfolio.total_value);
            println!("  Total Trades: {}", best_gen.best_portfolio.trades_count);
            println!("  Win Rate: {:.1}%", 
                if best_gen.best_portfolio.trades_count > 0 {
                    (best_gen.best_portfolio.winning_trades as f64 / best_gen.best_portfolio.trades_count as f64) * 100.0
                } else { 0.0 }
            );
            println!("  Max Drawdown: {:.2}%", best_gen.best_portfolio.max_drawdown * 100.0);
            println!("  Sharpe Ratio: {:.3}", best_gen.best_portfolio.sharpe_ratio());
        }
        
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("NEAT Trading Bot System");
    println!("======================");
    
    // Khởi tạo hệ thống
    let mut trading_system = NeatTradingSystem::new(
        50,     // 50 bots trong quần thể
        10000.0, // $10,000 ban đầu
        0.01,    // 1% mỗi lần giao dịch
        0.10,    // Giữ lại 10% elite
        0.20,    // 20% mutation
    );
    
    // Load dữ liệu BTC
    trading_system.load_market_data("examples/btc_ohlcv_history.csv")?;
    
    // Chạy evolution cho 30 thế hệ (giảm từ 50 để test nhanh hơn)
    trading_system.run_evolution(30)?;
    
    println!("\nEvolution completed successfully!");
    
    Ok(())
} 