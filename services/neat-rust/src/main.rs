use std::path::Path;
use neat_rust::{
    Neat,
    architecture::{network::Network, architect::Architect},
    utils::get_timestamp,
};
use neat_rust::neat::NeatOptions;

fn main() {
    println!("NEAT-RS XOR Demo with 2 approaches:\n");
    
    // Demo 1: Legacy approach with Perceptron Networks
    demo_perceptron_approach();
    
    // Demo 2: Full NEAT approach
    demo_neat_approach();
}

/// XOR evaluation function for network
fn xor_fitness(network: &Network) -> f64 {
    let xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];
    
    // Calculate error and convert to fitness with better dynamic range
    let error = network.evaluate(&xor_data);
    
    // Use exponential fitness function for better dynamic range
    (-error * 10.0).exp()
}

/// Demo using perceptron approach (legacy)
fn demo_perceptron_approach() {
    println!("=== APPROACH 1: Perceptron networks ===");
    println!("Running NEAT-RS XOR demo: 10 cá thể, 10 thế hệ\n");
    
    // Khởi tạo quần thể 10 mạng perceptron 2-2-1
    let mut population: Vec<Network> = (0..10)
        .map(|_| Architect::perceptron(&[2, 2, 1]))
        .collect();

    // Dữ liệu XOR
    let xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    for gen in 0..10 {
        println!("Thế hệ {}", gen + 1);
        // Đánh giá từng cá thể
        let mut scores: Vec<(usize, f64)> = population
            .iter()
            .enumerate()
            .map(|(i, net)| (i, net.evaluate(&xor_data)))
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Hiển thị top 3 cá thể tốt nhất
        for (idx, (i, score)) in scores.iter().take(3).enumerate() {
            println!("  Top {}: Cá thể {} - MSE = {:.6}", idx + 1, i + 1, score);
        }
        
        // Chọn lọc: giữ lại 5 cá thể tốt nhất, nhân bản và đột biến thành 10 cá thể mới
        let mut new_population = Vec::new();
        for &(i, _) in scores.iter().take(5) {
            new_population.push(population[i].clone());
            let mut mutated = population[i].clone();
            mutated.mutate();
            new_population.push(mutated);
        }
        
        // Huấn luyện từng cá thể
        for net in new_population.iter_mut() {
            net.train(&xor_data);
        }
        population = new_population;
    }
    
    // Test kết quả cuối cùng với cá thể tốt nhất
    println!("\n=== KẾT QUẢ CUỐI CÙNG PERCEPTRON ===");
    let final_scores: Vec<(usize, f64)> = population
        .iter()
        .enumerate()
        .map(|(i, net)| (i, net.evaluate(&xor_data)))
        .collect();
    
    let best_idx = final_scores.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap().0;
    
    let best_network = &population[best_idx];
    println!("Cá thể tốt nhất: #{} với MSE = {:.6}", best_idx + 1, best_network.evaluate(&xor_data));
    
    println!("\nTest XOR với cá thể tốt nhất:");
    for (input, expected) in &xor_data {
        let output = best_network.forward(&input[..]);
        let predicted = output[0];
        let rounded = if predicted > 0.5 { 1.0 } else { 0.0 };
        println!("Input: [{:.0}, {:.0}] -> Output: {:.4} -> Rounded: {:.0} (Expected: {:.0})", 
                input[0], input[1], predicted, rounded, expected);
    }
    
    // Save best network
    let timestamp = get_timestamp();
    let save_path = Path::new("./models").join(format!("xor_best_perceptron_{}.json", timestamp));
    
    if let Err(e) = best_network.save_to_json(&save_path) {
        println!("Không thể lưu model: {}", e);
    } else {
        println!("\nĐã lưu model tốt nhất tại: {:?}", save_path);
    }
}

/// Demo using full NEAT approach
fn demo_neat_approach() {
    println!("\n\n=== APPROACH 2: Full NEAT implementation ===");
    
    // Configure NEAT options
    let mut options = NeatOptions::default();
    options.popsize = 50;
    options.elitism = 5;
    options.mutation_rate = 0.7;  // Higher mutation rate
    options.mutation_amount = 0.5; // Smaller mutation amount
    options.fitness_population = false;
    options.reset_on_mutation = false; // Don't reset score after mutation
    
    // Create NEAT instance with XOR fitness function
    let mut neat = Neat::new(2, 1, xor_fitness, Some(options));
    
    println!("Running NEAT with population size {}", neat.options.popsize);
    
    // Run for 20 generations
    for i in 0..20 {
        match neat.evolve() {
            Ok(_best_score) => {
                let stats = neat.get_stats();
                println!(
                    "Gen {}: Best={:.4}, Avg={:.4}, Min={:.4}",
                    i + 1,
                    stats.max_score,
                    stats.avg_score,
                    stats.min_score
                );
            },
            Err(e) => {
                println!("Error during evolution: {:?}", e);
                break;
            }
        }
        
        // Stop if we found a good solution
        if let Some(best) = neat.get_best_network() {
            if let Some(score) = best.score {
                if score > 0.99 {
                    println!("Tìm thấy giải pháp tối ưu sau {} thế hệ!", i + 1);
                    break;
                }
            }
        }
    }
    
    // Test best network with XOR
    if let Some(best) = neat.get_best_network() {
        println!("\n=== KẾT QUẢ CUỐI CÙNG NEAT ===");
        println!("Mạng tốt nhất có fitness = {:.6}", best.score.unwrap_or(0.0));
        
        let xor_data = [
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];
        
        println!("\nTest XOR với mạng tốt nhất:");
        for (input, expected) in &xor_data {
            let output = best.forward(&input[..]);
            let predicted = output[0];
            let rounded = if predicted > 0.5 { 1.0 } else { 0.0 };
            println!(
                "Input: [{:.0}, {:.0}] -> Output: {:.4} -> Rounded: {:.0} (Expected: {:.0})", 
                input[0], input[1], predicted, rounded, expected
            );
        }
        
        // Save best network
        let timestamp = get_timestamp();
        let save_path = Path::new("./models").join(format!("xor_best_neat_{}.json", timestamp));
        
        if let Err(e) = best.save_to_json(&save_path) {
            println!("Không thể lưu model: {}", e);
        } else {
            println!("\nĐã lưu model tốt nhất tại: {:?}", save_path);
        }
        
        // Save entire NEAT state
        let neat_save_path = Path::new("./models").join(format!("neat_state_{}.json", timestamp));
        
        if let Err(e) = neat.save_to_json(&neat_save_path) {
            println!("Không thể lưu trạng thái NEAT: {}", e);
        } else {
            println!("Đã lưu trạng thái NEAT tại: {:?}", neat_save_path);
        }
    } else {
        println!("Không tìm được giải pháp tốt!");
    }
    
    println!("\nNEAT-RS đã hoàn thiện với đầy đủ chức năng!");
}
