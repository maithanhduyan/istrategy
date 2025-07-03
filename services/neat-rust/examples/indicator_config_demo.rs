/**
 * Demo: Gold NEAT Trading Bot với Dynamic Indicator Configuration
 * 
 * File này demo cách sử dụng hệ thống indicator configuration để:
 * 1. Bật/tắt các indicators một cách linh hoạt
 * 2. So sánh performance giữa các strategy khác nhau
 * 3. Load/save config từ JSON files
 * 4. Thay đổi config trong runtime
 */

use std::path::Path;
use serde_json;

// Import từ main module
use gold_neat_trading_bot::{
    IndicatorConfig, 
    NeatGoldTradingSystem
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DEMO: Dynamic Indicator Configuration ===\n");
    
    // 1. Demo các strategy có sẵn
    demo_predefined_strategies()?;
    
    // 2. Demo tạo custom strategy
    demo_custom_strategy()?;
    
    // 3. Demo load/save config
    demo_config_persistence()?;
    
    // 4. Demo so sánh strategies (chạy ngắn)
    demo_strategy_comparison()?;
    
    Ok(())
}

fn demo_predefined_strategies() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. === PREDEFINED STRATEGIES ===");
    
    let strategies = vec![
        ("Basic", IndicatorConfig::basic_strategy()),
        ("Advanced", IndicatorConfig::advanced_strategy()),
        ("Momentum", IndicatorConfig::momentum_strategy()),
        ("Trend Following", IndicatorConfig::trend_following_strategy()),
    ];
    
    for (name, config) in strategies {
        println!("Strategy: {}", name);
        println!("  Features: {}", config.calculate_feature_count());
        println!("  Indicators: {:?}", config.get_active_indicators());
        println!("  Lookback Days: {}", config.lookback_days);
        println!();
    }
    
    Ok(())
}

fn demo_custom_strategy() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. === CUSTOM STRATEGY ===");
    
    // Tạo strategy custom từ string
    let custom_config = IndicatorConfig::from_strategy_name("momentum");
    println!("Custom Strategy from string 'momentum':");
    println!("  Features: {}", custom_config.calculate_feature_count());
    println!("  Indicators: {:?}", custom_config.get_active_indicators());
    
    // Tạo strategy từ indicators list
    let indicators = vec!["ma_cross", "rsi", "momentum"];
    let custom_config2 = IndicatorConfig::from_indicators(indicators, 7);
    println!("\nCustom Strategy from indicators list:");
    println!("  Features: {}", custom_config2.calculate_feature_count());
    println!("  Indicators: {:?}", custom_config2.get_active_indicators());
    println!();
    
    Ok(())
}

fn demo_config_persistence() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. === CONFIG PERSISTENCE ===");
    
    // Save config
    let config = IndicatorConfig::momentum_strategy();
    config.save_to_file("examples/strategy_configs/demo_momentum.json")?;
    
    // Load config
    if Path::new("examples/strategy_configs/basic_strategy.json").exists() {
        let loaded_config = IndicatorConfig::load_from_file("examples/strategy_configs/basic_strategy.json")?;
        println!("Loaded config from file:");
        println!("  Features: {}", loaded_config.calculate_feature_count());
        println!("  Indicators: {:?}", loaded_config.get_active_indicators());
    }
    println!();
    
    Ok(())
}

fn demo_strategy_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. === STRATEGY COMPARISON (Quick Run) ===");
    
    let strategies = vec![
        ("Basic", IndicatorConfig::basic_strategy()),
        ("Momentum", IndicatorConfig::momentum_strategy()),
    ];
    
    for (name, config) in strategies {
        println!("Testing strategy: {}", name);
        
        let mut system = NeatGoldTradingSystem::new_with_config(
            10,      // Smaller population for demo
            10000.0,
            0.02,
            0.10,
            0.20,
            config,
        );
        
        system.load_gold_market_data("examples/gold_ohlcv_history.csv")?;
        
        println!("  {}", system.get_strategy_info());
        println!("  Active indicators: {:?}", system.indicator_config.get_active_indicators());
        
        // Quick test - chỉ chạy 3 generations
        system.run_evolution(3)?;
        println!();
    }
    
    Ok(())
}
