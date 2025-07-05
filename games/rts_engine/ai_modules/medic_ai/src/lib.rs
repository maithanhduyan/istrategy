use shared::*;
use std::collections::HashMap;

/// Medic AI Implementation  
/// Medics focus on healing allies and providing support
pub struct MedicAi {
    unit_id: u32,
    healing_range: f32,
    healing_power: f32,
    last_heal_time: f32,
    heal_cooldown: f32,
    follow_target: Option<u32>,
    safe_distance: f32,
    state_data: HashMap<String, f32>,
}

impl MedicAi {
    pub fn new() -> Self {
        Self {
            unit_id: 0,
            healing_range: 6.0,
            healing_power: 15.0,
            last_heal_time: 0.0,
            heal_cooldown: 2.0, // Heal every 2 seconds
            follow_target: None,
            safe_distance: 10.0, // Keep distance from enemies
            state_data: HashMap::new(),
        }
    }

    fn find_most_wounded_ally(
        &self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
    ) -> Option<u32> {
        let mut most_wounded: Option<(u32, f32)> = None;

        for ally in &environment.nearby_allies {
            if ally.id == unit_state.id {
                continue; // Skip self
            }

            let distance = unit_state.position.distance(ally.position);
            if distance <= self.healing_range {
                let health_ratio = ally.health / ally.max_health;

                // Find ally with lowest health ratio
                if let Some((_, current_lowest)) = most_wounded {
                    if health_ratio < current_lowest {
                        most_wounded = Some((ally.id, health_ratio));
                    }
                } else {
                    most_wounded = Some((ally.id, health_ratio));
                }
            }
        }

        most_wounded.map(|(id, _)| id)
    }

    fn find_ally_needing_help(
        &self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
    ) -> Option<u32> {
        // Find allies that are damaged and within reasonable distance
        for ally in &environment.nearby_allies {
            if ally.id == unit_state.id {
                continue;
            }

            let health_ratio = ally.health / ally.max_health;
            let distance = unit_state.position.distance(ally.position);

            // Prioritize critically wounded allies within medium range
            if health_ratio < 0.5 && distance <= 20.0 {
                return Some(ally.id);
            }
        }

        None
    }

    fn should_retreat(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> bool {
        // Retreat if enemies are too close and there are no strong allies nearby
        let nearby_enemies = environment.nearby_enemies.len();
        let nearby_strong_allies = environment
            .nearby_allies
            .iter()
            .filter(|ally| {
                let distance = unit_state.position.distance(ally.position);
                ally.unit_type == UnitType::Tank && distance <= 15.0
            })
            .count();

        // If enemies present and no tank protection
        nearby_enemies > 0 && nearby_strong_allies == 0
    }

    fn find_safe_position(&self, unit_state: &UnitState, environment: &EnvironmentInfo) -> Vec3 {
        // Move towards the nearest tank ally for protection
        for ally in &environment.nearby_allies {
            if ally.unit_type == UnitType::Tank {
                let direction = ally.position - unit_state.position;
                if direction.length() > 5.0 {
                    return ally.position + direction.normalize() * -3.0; // Stay behind tank
                }
            }
        }

        // If no tank, move away from enemies
        let mut retreat_direction = Vec3::ZERO;
        for enemy in &environment.nearby_enemies {
            let direction = unit_state.position - enemy.position;
            retreat_direction += direction.normalize();
        }

        if retreat_direction.length() > 0.1 {
            unit_state.position + retreat_direction.normalize() * self.safe_distance
        } else {
            // Last resort: random retreat
            use std::f32::consts::TAU;
            let angle = fastrand::f32() * TAU;
            unit_state.position
                + Vec3::new(
                    angle.cos() * self.safe_distance,
                    0.0,
                    angle.sin() * self.safe_distance,
                )
        }
    }

    fn can_heal(&self, current_time: f32) -> bool {
        current_time - self.last_heal_time >= self.heal_cooldown
    }
}

impl UnitAiModule for MedicAi {
    fn name(&self) -> &'static str {
        "Medic AI"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn supported_unit_types(&self) -> Vec<UnitType> {
        vec![UnitType::Medic]
    }

    fn update(
        &mut self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
        delta_time: f32,
    ) -> Option<UnitCommand> {
        self.last_heal_time += delta_time;

        // Medic AI Priority:
        // 1. Retreat from enemies if no protection
        // 2. Heal critically wounded allies in range
        // 3. Move to wounded allies outside range
        // 4. Follow and support strongest ally
        // 5. Stay safe and avoid combat

        // Check if should retreat
        if self.should_retreat(unit_state, environment) {
            let safe_pos = self.find_safe_position(unit_state, environment);
            return Some(UnitCommand::Retreat {
                safe_position: safe_pos,
            });
        }

        // Check for healing opportunities
        if self.can_heal(self.last_heal_time) {
            if let Some(target_id) = self.find_most_wounded_ally(unit_state, environment) {
                self.last_heal_time = 0.0;
                return Some(UnitCommand::Heal { target_id });
            }
        }

        // Find allies that need help but are out of range
        if let Some(ally_id) = self.find_ally_needing_help(unit_state, environment) {
            // Find the ally's position
            for ally in &environment.nearby_allies {
                if ally.id == ally_id {
                    let distance = unit_state.position.distance(ally.position);
                    if distance > self.healing_range {
                        // Move closer to heal
                        let direction = ally.position - unit_state.position;
                        let target = unit_state.position
                            + direction.normalize() * (distance - self.healing_range + 1.0);
                        return Some(UnitCommand::MoveTo { target });
                    }
                }
            }
        }

        // Support behavior: follow strongest ally for protection
        match unit_state.state {
            UnitAiState::Idle => {
                // Find a tank to follow for protection
                for ally in &environment.nearby_allies {
                    if ally.unit_type == UnitType::Tank {
                        let distance = unit_state.position.distance(ally.position);
                        if distance > 8.0 {
                            // Follow at safe distance
                            let direction = ally.position - unit_state.position;
                            let follow_pos = ally.position + direction.normalize() * -5.0;
                            return Some(UnitCommand::Follow { target_id: ally.id });
                        }
                        break;
                    }
                }

                // If no specific ally to follow, stay in current position or move to center of group
                if !environment.nearby_allies.is_empty() {
                    let avg_pos = environment
                        .nearby_allies
                        .iter()
                        .fold(Vec3::ZERO, |acc, ally| acc + ally.position)
                        / environment.nearby_allies.len() as f32;

                    let distance = unit_state.position.distance(avg_pos);
                    if distance > 10.0 {
                        return Some(UnitCommand::MoveTo { target: avg_pos });
                    }
                }

                None
            }
            UnitAiState::Moving => {
                // Continue current movement, but prioritize healing if critical cases appear
                if let Some(target_id) = self.find_most_wounded_ally(unit_state, environment) {
                    // Check if ally is critically wounded
                    for ally in &environment.nearby_allies {
                        if ally.id == target_id && ally.health / ally.max_health < 0.3 {
                            if self.can_heal(self.last_heal_time) {
                                self.last_heal_time = 0.0;
                                return Some(UnitCommand::Heal { target_id });
                            }
                        }
                    }
                }
                None
            }
            UnitAiState::Healing => {
                // Continue healing
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
        self.follow_target = None;
        self.state_data.clear();

        println!("Medic AI initialized for unit {}", unit_state.id);
    }

    fn cleanup(&mut self, unit_id: u32) {
        println!("Medic AI cleanup for unit {}", unit_id);
        self.state_data.clear();
    }
}

/// Export functions for dynamic loading
#[no_mangle]
pub extern "C" fn create_ai_module() -> *mut dyn UnitAiModule {
    let medic_ai = MedicAi::new();
    Box::into_raw(Box::new(medic_ai))
}

#[no_mangle]
pub extern "C" fn get_module_metadata() -> AiModuleMetadata {
    AiModuleMetadata {
        name: "Medic AI".to_string(),
        version: "1.0.0".to_string(),
        supported_unit_types: vec![UnitType::Medic],
        description: "Support AI that heals wounded allies, stays safe, and follows strong units for protection.".to_string(),
        author: "RTS Engine Team".to_string(),
    }
}
