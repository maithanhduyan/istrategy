use neat_rust::architecture::architect::Architect;
use neat_rust::architecture::network::Network;

fn main() {
    println!("NEAT-RS XOR demo: 10 cá thể, 10 thế hệ");
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
        println!("\nThế hệ {}", gen + 1);
        // Đánh giá từng cá thể
        let mut scores: Vec<(usize, f64)> = population
            .iter()
            .enumerate()
            .map(|(i, net)| (i, net.evaluate(&xor_data)))
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Hiển thị top 3 cá thể tốt nhất
        for (idx, (i, score)) in scores.iter().take(3).enumerate() {
            println!("Top {}: Cá thể {} - MSE = {:.6}", idx + 1, i + 1, score);
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
    println!("\n=== KẾT QUẢ CUỐI CÙNG ===");
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
    println!("\nNEAT-RS XOR đã hoàn thiện với logic forward, mutate, train thực tế!");
}
