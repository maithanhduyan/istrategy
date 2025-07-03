/**
 * Strategy Comparison Tool - So sánh hiệu suất các strategy khác nhau
 * 
 * Tool này sẽ chạy nhanh với các strategy khác nhau để so sánh performance
 */

use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use serde::{Deserialize, Serialize};
use neat_rust::{
    architecture::network::Network,
    utils::get_timestamp,
};

mod gold_neat_trading_bot;
use gold_neat_trading_bot::*;

#[derive(Debug, Clone, Serialize)]
struct StrategyComparison {
    strategy_name: String,
    feature_count: usize,
    active_indicators: Vec<String>,
    best_roi: f64,
    best_fitness: f64,
    avg_roi: f64,
    best_win_rate: f64,
    final_value: f64,
    generations_tested: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== GOLD TRADING STRATEGY COMPARISON TOOL ===\n");
    
    let strategies = vec![
        ("Basic", IndicatorConfig::basic_strategy()),
        ("Advanced", IndicatorConfig::advanced_strategy()),
        ("Momentum", IndicatorConfig::momentum_strategy()),
        ("Trend_Following", IndicatorConfig::trend_following_strategy()),
    ];
    
    let mut comparison_results = Vec::new();
    
    println!("Testing {} strategies with 20 generations each...\n", strategies.len());
    
    for (name, config) in strategies {
        println!("=== Testing Strategy: {} ===", name);
        println!("Features: {}", config.calculate_feature_count());
        println!("Indicators: {:?}", config.get_active_indicators());
        
        let mut system = NeatGoldTradingSystem::new_with_config(
            30,      // Smaller population for faster comparison
            10000.0,
            0.02,
            0.10,
            0.20,
            config.clone(),
        );
        
        system.load_gold_market_data("examples/gold_ohlcv_history.csv")?;
        
        // Chạy với ít generations hơn để so sánh nhanh
        system.run_evolution(20)?;
        
        // Lấy kết quả tốt nhất
        if let Some(best_stats) = system.generation_stats.iter().max_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap()) {
            let result = StrategyComparison {
                strategy_name: name.to_string(),
                feature_count: config.calculate_feature_count(),
                active_indicators: config.get_active_indicators(),
                best_roi: best_stats.best_roi,
                best_fitness: best_stats.best_fitness,
                avg_roi: best_stats.avg_roi,
                best_win_rate: best_stats.best_win_rate,
                final_value: best_stats.best_portfolio.total_value,
                generations_tested: 20,
            };
            
            comparison_results.push(result);
            
            println!("Results:");
            println!("  Best ROI: {:.2}%", best_stats.best_roi * 100.0);
            println!("  Best Fitness: {:.4}", best_stats.best_fitness);
            println!("  Final Value: ${:.2}", best_stats.best_portfolio.total_value);
            println!("  Win Rate: {:.1}%", best_stats.best_win_rate * 100.0);
            println!("  Max Drawdown: {:.2}%", best_stats.best_portfolio.max_drawdown * 100.0);
        }
        
        println!();
    }
    
    // Lưu kết quả so sánh
    let timestamp = get_timestamp();
    let comparison_file = format!("models/strategy_comparison_{}.json", timestamp);
    let json = serde_json::to_string_pretty(&comparison_results)?;
    std::fs::write(&comparison_file, json)?;
    
    // In bảng so sánh cuối cùng
    println!("=== STRATEGY COMPARISON SUMMARY ===");
    println!("{:<15} {:<10} {:<8} {:<12} {:<10} {:<12}", 
             "Strategy", "Features", "ROI%", "Fitness", "Win%", "Final$");
    println!("{}", "-".repeat(80));
    
    // Sắp xếp theo fitness
    comparison_results.sort_by(|a, b| b.best_fitness.partial_cmp(&a.best_fitness).unwrap());
    
    for result in &comparison_results {
        println!("{:<15} {:<10} {:<8.1} {:<12.4} {:<10.1} {:<12.0}", 
                 result.strategy_name,
                 result.feature_count,
                 result.best_roi * 100.0,
                 result.best_fitness,
                 result.best_win_rate * 100.0,
                 result.final_value);
    }
    
    println!("\nBest Strategy: {}", comparison_results[0].strategy_name);
    println!("Comparison results saved to: {}", comparison_file);
    
    // Tạo recommendation
    println!("\n=== RECOMMENDATIONS ===");
    let best = &comparison_results[0];
    println!("🏆 Best Overall: {} (Fitness: {:.4})", best.strategy_name, best.best_fitness);
    
    let highest_roi = comparison_results.iter().max_by(|a, b| a.best_roi.partial_cmp(&b.best_roi).unwrap()).unwrap();
    println!("💰 Highest ROI: {} ({:.1}%)", highest_roi.strategy_name, highest_roi.best_roi * 100.0);
    
    let best_win_rate = comparison_results.iter().max_by(|a, b| a.best_win_rate.partial_cmp(&b.best_win_rate).unwrap()).unwrap();
    println!("🎯 Best Win Rate: {} ({:.1}%)", best_win_rate.strategy_name, best_win_rate.best_win_rate * 100.0);
    
    let most_efficient = comparison_results.iter().min_by_key(|r| r.feature_count).unwrap();
    println!("⚡ Most Efficient: {} ({} features)", most_efficient.strategy_name, most_efficient.feature_count);
    
    Ok(())
}
