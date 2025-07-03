/**
 * Demo Runtime Strategy Switching
 * 
 * Thể hiện cách thay đổi indicator configuration trong runtime
 */

use std::collections::HashMap;

mod gold_neat_trading_bot;
use gold_neat_trading_bot::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RUNTIME STRATEGY SWITCHING DEMO ===\n");
    
    // Khởi tạo với basic strategy
    let mut system = NeatGoldTradingSystem::new_with_config(
        10,      // Small population for demo
        10000.0,
        0.02,
        0.10,
        0.20,
        IndicatorConfig::basic_strategy(),
    );
    
    system.load_gold_market_data("examples/gold_ohlcv_history.csv")?;
    
    println!("1. Initial Strategy:");
    println!("   {}", system.get_strategy_info());
    println!("   Indicators: {:?}\n", system.indicator_config.get_active_indicators());
    
    // Demo switching strategies
    let strategies = [
        ("Momentum", IndicatorConfig::momentum_strategy()),
        ("Advanced", IndicatorConfig::advanced_strategy()),
        ("Trend Following", IndicatorConfig::trend_following_strategy()),
        ("Custom Minimal", create_minimal_strategy()),
        ("Custom Volatility Focus", create_volatility_strategy()),
    ];
    
    let mut performance_results = HashMap::new();
    
    for (name, config) in strategies.iter() {
        println!("2. Switching to {} Strategy:", name);
        
        // Thay đổi strategy
        system.set_indicator_config(config.clone());
        
        println!("   {}", system.get_strategy_info());
        println!("   Indicators: {:?}", system.indicator_config.get_active_indicators());
        
        // Quick test - 3 generations
        system.run_evolution(3)?;
        
        // Lưu kết quả
        if let Some(best_stats) = system.generation_stats.last() {
            performance_results.insert(name.clone(), (
                best_stats.best_roi,
                best_stats.best_fitness,
                best_stats.best_win_rate,
                config.calculate_feature_count(),
            ));
            
            println!("   Quick Results: ROI {:.1}%, Fitness {:.3}, Win Rate {:.1}%", 
                     best_stats.best_roi * 100.0,
                     best_stats.best_fitness,
                     best_stats.best_win_rate * 100.0);
        }
        
        // Clear previous stats for next test
        system.generation_stats.clear();
        system.current_generation = 0;
        
        println!();
    }
    
    // Summary
    println!("=== PERFORMANCE SUMMARY ===");
    println!("{:<20} {:<8} {:<10} {:<8} {:<10}", "Strategy", "Features", "ROI%", "Fitness", "Win%");
    println!("{}", "-".repeat(65));
    
    let mut sorted_results: Vec<_> = performance_results.into_iter().collect();
    sorted_results.sort_by(|a, b| b.1.1.partial_cmp(&a.1.1).unwrap()); // Sort by fitness
    
    for (name, (roi, fitness, win_rate, features)) in sorted_results {
        println!("{:<20} {:<8} {:<10.1} {:<8.3} {:<10.1}", 
                 name, features, roi * 100.0, fitness, win_rate * 100.0);
    }
    
    println!("\n=== STRATEGY RECOMMENDATIONS ===");
    println!("🎯 For Quick Decisions: Use strategies with fewer features (< 15)");
    println!("💪 For Max Performance: Use Advanced strategy with all indicators");
    println!("⚡ For Speed: Use Basic or Custom Minimal strategies");
    println!("📈 For Trending Markets: Use Trend Following strategy");
    println!("🌊 For Volatile Markets: Use Momentum or Volatility Focus strategies");
    
    Ok(())
}

fn create_minimal_strategy() -> IndicatorConfig {
    IndicatorConfig {
        enable_ma_30: false,
        enable_ma_cross: true,   // Chỉ MA Cross
        enable_rsi: false,
        enable_bollinger_bands: false,
        enable_volatility: false,
        enable_price_momentum: false,
        enable_volume_trend: false,
        enable_basic_ohlcv: true,
        lookback_days: 2,        // Minimal lookback
    }
}

fn create_volatility_strategy() -> IndicatorConfig {
    IndicatorConfig {
        enable_ma_30: false,
        enable_ma_cross: false,
        enable_rsi: true,
        enable_bollinger_bands: true,
        enable_volatility: true,     // Focus on volatility
        enable_price_momentum: true,
        enable_volume_trend: true,
        enable_basic_ohlcv: true,
        lookback_days: 3,
    }
}
