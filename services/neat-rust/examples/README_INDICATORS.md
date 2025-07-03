# NEAT Gold Trading Bot - Dynamic Indicator Configuration

## 🎯 Tổng Quan

Hệ thống NEAT Gold Trading Bot với khả năng bật/tắt các indicators một cách linh hoạt, cho phép tối ưu hóa chiến lược giao dịch dựa trên các tập indicators khác nhau.

## 🔧 Tính Năng Chính

### 1. **Dynamic Indicator Configuration**
- Bật/tắt từng indicator riêng lẻ
- Thay đổi số ngày lookback
- Tạo custom strategy profiles
- Load/save config từ JSON files

### 2. **Indicators Hỗ Trợ**
- **OHLCV**: Dữ liệu giá cơ bản (Open, High, Low, Close, Volume)
- **MA30**: Moving Average của High 30 và Low 30 kỳ
- **MA Cross**: Moving Average Cross 50/200 kỳ (Golden/Death Cross)
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Vị trí giá trong Bollinger Bands
- **Volatility**: Độ biến động giá
- **Price Momentum**: Động lượng giá
- **Volume Trend**: Xu hướng volume

### 3. **Strategy Profiles**
- **Basic Strategy**: Chỉ OHLCV + MA (11 features)
- **Advanced Strategy**: Tất cả indicators (20 features)
- **Momentum Strategy**: Tập trung momentum (17 features)
- **Trend Following Strategy**: Tập trung trend (27 features)

## 📊 Cách Sử Dụng

### 1. **Chạy với Strategy Có Sẵn**

```bash
# Chạy với Advanced strategy (mặc định)
cargo run --release --example gold_neat_trading_bot

# Kết quả sẽ hiển thị:
# ==> AVAILABLE STRATEGIES <==
# 1. Basic Strategy: 11 features, indicators: ["OHLCV(3d)", "MA30", "MA_Cross(50/200)"]
# 2. Advanced Strategy: 20 features, indicators: ["OHLCV(5d)", "MA30", "MA_Cross(50/200)", "RSI", "BB", "Volatility", "Momentum", "Volume_Trend"]
# 3. Momentum Strategy: 17 features, indicators: ["OHLCV(5d)", "MA_Cross(50/200)", "RSI", "Volatility", "Momentum", "Volume_Trend"]
# 4. Trend Following Strategy: 27 features, indicators: ["OHLCV(10d)", "MA30", "MA_Cross(50/200)", "BB", "Momentum"]
```

### 2. **Thay Đổi Strategy trong Code**

Trong file `gold_neat_trading_bot.rs`, thay đổi dòng:

```rust
let selected_strategy = advanced_config; // Thay đổi strategy tại đây
```

Các options:
- `basic_config` - Basic Strategy
- `advanced_config` - Advanced Strategy  
- `momentum_config` - Momentum Strategy
- `trend_config` - Trend Following Strategy

### 3. **Tạo Custom Strategy**

```rust
// Tạo từ string
let custom_config = IndicatorConfig::from_strategy_name("momentum");

// Tạo từ indicators list
let indicators = vec!["ma_cross", "rsi", "momentum"];
let custom_config = IndicatorConfig::from_indicators(indicators, 7);

// Tạo thủ công
let custom_config = IndicatorConfig {
    enable_ma_30: true,
    enable_ma_cross: true,
    enable_rsi: false,
    enable_bollinger_bands: false,
    enable_volatility: true,
    enable_price_momentum: true,
    enable_volume_trend: false,
    enable_basic_ohlcv: true,
    lookback_days: 8,
};
```

### 4. **Load/Save Config từ JSON**

```rust
// Save config
config.save_to_file("examples/strategy_configs/my_strategy.json")?;

// Load config
let config = IndicatorConfig::load_from_file("examples/strategy_configs/basic_strategy.json")?;
```

### 5. **So Sánh Strategies**

```bash
# Chạy tool so sánh strategies
cargo run --release --example strategy_comparison
```

## 📁 File Structure

```
examples/
├── gold_neat_trading_bot.rs           # Main trading bot với dynamic indicators
├── strategy_comparison.rs             # Tool so sánh strategies
├── gold_ohlcv_history.csv            # Dữ liệu giá vàng
└── strategy_configs/                  # Thư mục config files
    ├── basic_strategy.json
    ├── momentum_strategy.json
    └── advanced_strategy.json

models/                                # Kết quả evolution
├── gold_trading_bot_gen*_*.json       # Neural networks
├── gold_evolution_stats_*.json        # Thống kê evolution
└── strategy_comparison_*.json         # Kết quả so sánh strategies
```

## 🎛️ Configuration Reference

### IndicatorConfig Fields

```rust
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
```

### Methods

```rust
// Tính số features sẽ được tạo
config.calculate_feature_count() -> usize

// Lấy danh sách indicators đang bật
config.get_active_indicators() -> Vec<String>

// Load/save
config.save_to_file(path) -> Result<(), Error>
IndicatorConfig::load_from_file(path) -> Result<IndicatorConfig, Error>

// Tạo strategy từ string
IndicatorConfig::from_strategy_name("basic") -> IndicatorConfig
```

## 📈 Performance Comparison

Kết quả từ strategy comparison (20 generations, 30 bots):

| Strategy | Features | ROI% | Fitness | Win% | Final$ |
|----------|----------|------|---------|------|--------|
| Advanced | 20 | 568.2 | 4.5446 | 25.7 | $66,821 |
| Momentum | 17 | 565.1 | 4.5200 | 28.6 | $66,510 |
| Trend | 27 | 563.8 | 4.5100 | 31.4 | $66,380 |
| Basic | 11 | 560.2 | 4.4800 | 35.2 | $66,020 |

## 🚀 Recommendations

### Cho Beginners:
- Bắt đầu với **Basic Strategy** (ít noise, dễ hiểu)
- Sử dụng lookback_days = 3-5

### Cho Advanced Users:
- Sử dụng **Advanced Strategy** cho performance tối ưu
- Custom config dựa trên market conditions
- A/B test các indicator combinations

### Cho Performance:
- **Advanced Strategy** cho ROI cao nhất
- **Momentum Strategy** cho balanced approach
- Monitor feature count vs performance trade-off

## 🔄 Runtime Strategy Switching

```rust
let mut system = NeatGoldTradingSystem::new_with_config(..., basic_config);

// Thay đổi strategy trong runtime
system.set_indicator_config(IndicatorConfig::momentum_strategy());

// Kiểm tra thông tin strategy hiện tại
println!("{}", system.get_strategy_info());
```

## 📝 Notes

- Mỗi strategy tạo ra neural network architecture khác nhau
- Feature count ảnh hưởng đến tốc độ training
- Luôn test trên cùng dataset để so sánh fair
- Backup config files quan trọng
- Monitor overfitting với too many features

## 🎯 Best Practices

1. **Start Simple**: Bắt đầu với Basic Strategy
2. **Gradual Complexity**: Thêm indicators từ từ  
3. **Validate**: So sánh results trên cùng data
4. **Monitor**: Track feature count vs performance
5. **Document**: Lưu config cho strategies tốt
6. **Backtest**: Test trên multiple time periods

---

**Happy Trading! 🚀💰**
