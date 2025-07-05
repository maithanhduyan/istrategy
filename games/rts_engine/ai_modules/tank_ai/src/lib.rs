use shared::*;
use std::collections::HashMap;

/// Tank AI Implementation
/// Tanks are aggressive units that seek and destroy enemies
pub struct TankAi {
    unit_id: u32,
    last_target_search: f32,
    search_interval: f32,
    patrol_points: Vec<Vec3>,
    current_patrol_index: usize,
    aggro_range: f32,
    attack_range: f32,
    state_data: HashMap<String, f32>,
}

impl TankAi {
    pub fn new() -> Self {
        Self {
            unit_id: 0,
            last_target_search: 0.0,
            search_interval: 1.0, // Search for targets every second
            patrol_points: Vec::new(),
            current_patrol_index: 0,
            aggro_range: 15.0,
            attack_range: 8.0,
            state_data: HashMap::new(),
        }
    }

    fn find_nearest_enemy(
        &self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
    ) -> Option<u32> {
        let mut nearest_enemy = None;
        let mut nearest_distance = f32::MAX;

        for enemy in &environment.nearby_enemies {
            let distance = unit_state.position.distance(enemy.position);
            if distance < self.aggro_range && distance < nearest_distance {
                nearest_distance = distance;
                nearest_enemy = Some(enemy.id);
            }
        }

        nearest_enemy
    }

    fn should_retreat(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> bool {
        // Retreat if health is low and outnumbered
        let health_ratio = unit_state.health / unit_state.max_health;
        let enemy_count = environment.nearby_enemies.len();
        let ally_count = environment.nearby_allies.len();

        health_ratio < 0.3 && enemy_count > ally_count + 1
    }

    fn find_safe_position(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> Vec3 {
        // Simple retreat logic: move away from enemies towards allies
        let mut retreat_direction = Vec3::ZERO;

        // Move away from enemies
        for enemy in &environment.nearby_enemies {
            let direction = unit_state.position - enemy.position;
            retreat_direction += direction.normalize() * 2.0;
        }

        // Move towards allies
        for ally in &environment.nearby_allies {
            let direction = ally.position - unit_state.position;
            retreat_direction += direction.normalize();
        }

        unit_state.position + retreat_direction.normalize() * 20.0
    }

    fn get_patrol_target(&mut self, unit_state: &UnitState) -> Option<Vec3> {
        if self.patrol_points.is_empty() {
            // Generate default patrol points around spawn
            let spawn_pos = unit_state.position;
            self.patrol_points = vec![
                spawn_pos + Vec3::new(10.0, 0.0, 0.0),
                spawn_pos + Vec3::new(0.0, 0.0, 10.0),
                spawn_pos + Vec3::new(-10.0, 0.0, 0.0),
                spawn_pos + Vec3::new(0.0, 0.0, -10.0),
            ];
        }

        if let Some(current_target) = self.patrol_points.get(self.current_patrol_index) {
            // Check if reached current patrol point
            if unit_state.position.distance(*current_target) < 2.0 {
                self.current_patrol_index =
                    (self.current_patrol_index + 1) % self.patrol_points.len();
            }
            Some(*current_target)
        } else {
            None
        }
    }
}

impl UnitAiModule for TankAi {
    fn name(&self) -> &'static str {
        "Tank AI"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn supported_unit_types(&self) -> Vec<UnitType> {
        vec![UnitType::Tank]
    }

    fn update(
        &mut self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
        delta_time: f32,
    ) -> Option<UnitCommand> {
        self.last_target_search += delta_time;

        // Tank AI Logic:
        // 1. If low health and outnumbered -> Retreat
        // 2. If enemy in range -> Attack
        // 3. If no enemies -> Patrol/Idle

        // Check if should retreat
        if self.should_retreat(unit_state, environment) {
            let safe_pos = self.find_safe_position(unit_state, environment);
            return Some(UnitCommand::Retreat {
                safe_position: safe_pos,
            });
        }

        // Search for enemies periodically
        if self.last_target_search >= self.search_interval {
            if let Some(enemy_id) = self.find_nearest_enemy(unit_state, environment) {
                self.last_target_search = 0.0;

                // Find the enemy unit
                for enemy in &environment.nearby_enemies {
                    if enemy.id == enemy_id {
                        let distance = unit_state.position.distance(enemy.position);

                        if distance <= self.attack_range {
                            // In attack range, attack directly
                            return Some(UnitCommand::AttackUnit {
                                target_id: enemy_id,
                            });
                        } else {
                            // Move to attack range
                            return Some(UnitCommand::AttackMove {
                                target: enemy.position,
                            });
                        }
                    }
                }
            }
            self.last_target_search = 0.0;
        }

        // No enemies found, patrol or idle
        match unit_state.state {
            UnitAiState::Idle => {
                if let Some(patrol_target) = self.get_patrol_target(unit_state) {
                    Some(UnitCommand::MoveTo {
                        target: patrol_target,
                    })
                } else {
                    None
                }
            }
            UnitAiState::Moving => {
                // Continue current movement, but check for new orders next update
                None
            }
            _ => {
                // For other states, let the current action complete
                None
            }
        }
    }

    fn initialize(&mut self, unit_state: &UnitState) {
        self.unit_id = unit_state.id;
        self.patrol_points.clear();
        self.current_patrol_index = 0;
        self.state_data.clear();

        println!("Tank AI initialized for unit {}", unit_state.id);
    }

    fn cleanup(&mut self, unit_id: u32) {
        println!("Tank AI cleanup for unit {}", unit_id);
        self.state_data.clear();
    }
}

/// Export functions for dynamic loading
#[no_mangle]
pub extern "C" fn create_ai_module() -> *mut dyn UnitAiModule {
    let tank_ai = TankAi::new();
    Box::into_raw(Box::new(tank_ai))
}

#[no_mangle]
pub extern "C" fn get_module_metadata() -> AiModuleMetadata {
    AiModuleMetadata {
        name: "Tank AI".to_string(),
        version: "1.0.0".to_string(),
        supported_unit_types: vec![UnitType::Tank],
        description:
            "Aggressive AI for tank units. Seeks and destroys enemies, retreats when low health."
                .to_string(),
        author: "RTS Engine Team".to_string(),
    }
}
