use shared::*;
use std::collections::HashMap;

/// Scout AI Implementation
/// Scouts are fast, fragile units focused on reconnaissance and intelligence gathering
pub struct ScoutAi {
    unit_id: u32,
    scouting_points: Vec<Vec3>,
    current_scout_index: usize,
    last_enemy_report: f32,
    report_interval: f32,
    flee_distance: f32,
    safe_distance: f32,
    state_data: HashMap<String, f32>,
    discovered_enemies: Vec<u32>,
}

impl ScoutAi {
    pub fn new() -> Self {
        Self {
            unit_id: 0,
            scouting_points: Vec::new(),
            current_scout_index: 0,
            last_enemy_report: 0.0,
            report_interval: 2.0,
            flee_distance: 12.0, // Start fleeing when enemy is this close
            safe_distance: 20.0, // Consider safe when enemy is this far
            state_data: HashMap::new(),
            discovered_enemies: Vec::new(),
        }
    }

    fn should_flee(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> bool {
        // Scout should flee from any enemy within flee distance
        for enemy in &environment.nearby_enemies {
            if unit_state.position.distance(enemy.position) < self.flee_distance {
                return true;
            }
        }
        false
    }

    fn find_flee_direction(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> Vec3 {
        let mut flee_direction = Vec3::ZERO;
        
        // Calculate direction away from all nearby enemies
        for enemy in &environment.nearby_enemies {
            let direction = unit_state.position - enemy.position;
            let distance = direction.length();
            if distance > 0.0 {
                // Weight by inverse distance - closer enemies have more influence
                let weight = 1.0 / distance.max(0.1);
                flee_direction += direction.normalize() * weight;
            }
        }

        if flee_direction.length() > 0.0 {
            unit_state.position + flee_direction.normalize() * self.safe_distance
        } else {
            // Fallback: move to random safe position
            self.get_random_scouting_position(unit_state)
        }
    }

    fn get_next_scouting_point(&mut self, unit_state: &UnitState) -> Vec3 {
        if self.scouting_points.is_empty() {
            self.generate_scouting_points(unit_state);
        }

        if let Some(current_point) = self.scouting_points.get(self.current_scout_index) {
            // Check if reached current scouting point
            if unit_state.position.distance(*current_point) < 3.0 {
                self.current_scout_index = (self.current_scout_index + 1) % self.scouting_points.len();
                if let Some(next_point) = self.scouting_points.get(self.current_scout_index) {
                    *next_point
                } else {
                    *current_point
                }
            } else {
                *current_point
            }
        } else {
            self.get_random_scouting_position(unit_state)
        }
    }

    fn generate_scouting_points(&mut self, unit_state: &UnitState) {
        let center = unit_state.position;
        let radius = 30.0;
        
        // Generate points in a circular pattern for scouting
        for i in 0..8 {
            let angle = (i as f32) * (2.0 * std::f32::consts::PI / 8.0);
            let x = center.x + radius * angle.cos();
            let z = center.z + radius * angle.sin();
            self.scouting_points.push(Vec3::new(x, center.y, z));
        }
    }

    fn get_random_scouting_position(&self, unit_state: &UnitState) -> Vec3 {
        use std::f32::consts::TAU;
        
        let angle = fastrand::f32() * TAU;
        let distance = 15.0 + fastrand::f32() * 25.0; // 15-40 units away
        
        Vec3::new(
            unit_state.position.x + angle.cos() * distance,
            unit_state.position.y,
            unit_state.position.z + angle.sin() * distance,
        )
    }

    fn report_enemy_positions(&mut self, environment: &EnvironmentInfo) {
        // In a real implementation, this would send intel to team/commander
        for enemy in &environment.nearby_enemies {
            if !self.discovered_enemies.contains(&enemy.id) {
                self.discovered_enemies.push(enemy.id);
                println!("Scout {} discovered enemy {} at {:?}", 
                    self.unit_id, enemy.id, enemy.position);
            }
        }
    }

    fn is_safe_position(&self, position: Vec3, environment: &EnvironmentInfo) -> bool {
        for enemy in &environment.nearby_enemies {
            if position.distance(enemy.position) < self.safe_distance {
                return false;
            }
        }
        true
    }
}

impl UnitAiModule for ScoutAi {
    fn name(&self) -> &'static str {
        "Scout AI"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn supported_unit_types(&self) -> Vec<UnitType> {
        vec![UnitType::Scout]
    }

    fn update(
        &mut self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
        delta_time: f32,
    ) -> Option<UnitCommand> {
        self.last_enemy_report += delta_time;

        // Scout AI Logic:
        // 1. If enemies nearby -> Flee to safe distance
        // 2. Report enemy positions periodically
        // 3. Continue scouting patrol when safe

        // Check if need to flee
        if self.should_flee(unit_state, environment) {
            let flee_position = self.find_flee_direction(unit_state, environment);
            return Some(UnitCommand::MoveTo { target: flee_position });
        }

        // Report enemy positions periodically
        if self.last_enemy_report >= self.report_interval {
            self.report_enemy_positions(environment);
            self.last_enemy_report = 0.0;
        }

        // Continue scouting mission
        match unit_state.state {
            UnitAiState::Idle => {
                let next_point = self.get_next_scouting_point(unit_state);
                Some(UnitCommand::MoveTo { target: next_point })
            }
            UnitAiState::Moving => {
                // Check if current target is still safe
                if let Some(target) = unit_state.target_position {
                    if !self.is_safe_position(target, environment) {
                        // Current target is no longer safe, find new one
                        let safe_point = self.get_random_scouting_position(unit_state);
                        Some(UnitCommand::MoveTo { target: safe_point })
                    } else {
                        None // Continue current movement
                    }
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn initialize(&mut self, unit_state: &UnitState) {
        self.unit_id = unit_state.id;
        self.scouting_points.clear();
        self.current_scout_index = 0;
        self.state_data.clear();
        self.discovered_enemies.clear();
        
        println!("Scout AI initialized for unit {}", unit_state.id);
    }

    fn cleanup(&mut self, unit_id: u32) {
        println!("Scout AI cleanup for unit {}", unit_id);
        self.state_data.clear();
        self.discovered_enemies.clear();
    }
}

/// Export functions for dynamic loading
#[no_mangle]
pub extern "C" fn create_ai_module() -> *mut dyn UnitAiModule {
    let scout_ai = ScoutAi::new();
    Box::into_raw(Box::new(scout_ai))
}

#[no_mangle]
pub extern "C" fn get_module_metadata() -> AiModuleMetadata {
    AiModuleMetadata {
        name: "Scout AI".to_string(),
        version: "1.0.0".to_string(),
        supported_unit_types: vec![UnitType::Scout],
        description: "Reconnaissance AI for scout units. Explores map, gathers intel, avoids combat.".to_string(),
        author: "RTS Engine Team".to_string(),
    }
}
