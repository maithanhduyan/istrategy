use bevy::prelude::*;
use bevy::input::mouse::{MouseButton, MouseButtonInput};
use bevy::window::PrimaryWindow;
use shared::*;
use crate::components::*;
use crate::ai_loader::AiModuleManager;
use crate::GameState;

/// System để khởi tạo AI modules khi game start
pub fn update_ai_modules(
    mut ai_manager: ResMut<AiModuleManager>,
    query: Query<(Entity, &UnitComponent), Added<UnitComponent>>,
) {
    // Load tất cả modules nếu chưa load
    if ai_manager.get_loaded_modules().is_empty() {
        if let Err(e) = ai_manager.load_all_modules() {
            error!("Failed to load AI modules: {}", e);
        }
    }

    // Initialize AI cho units mới spawn
    for (entity, unit) in query.iter() {
        if let Some(ref module_name) = unit.ai_module_name {
            if let Err(e) = ai_manager.create_ai_instance(unit.state.id, module_name) {
                error!("Failed to create AI instance for unit {}: {}", unit.state.id, e);
            } else {
                ai_manager.initialize_ai_instance(unit.state.id, module_name, &unit.state);
                info!("Initialized AI for unit {} with module {}", unit.state.id, module_name);
            }
        }
    }
}

/// System để update AI logic cho các units
pub fn update_unit_ai(
    mut ai_manager: ResMut<AiModuleManager>,
    mut unit_query: Query<(&mut UnitComponent, &mut AiStateComponent, &Transform)>,
    all_units_query: Query<(&UnitComponent, &Transform)>,
    time: Res<Time>,
) {
    let current_time = time.elapsed_secs();
    
    for (mut unit, mut ai_state, transform) in unit_query.iter_mut() {
        // Check if it's time to update AI
        if current_time - ai_state.last_update < ai_state.update_interval {
            continue;
        }

        if let Some(module_name) = unit.ai_module_name.clone() {
            // Update unit state với transform hiện tại
            unit.state.position = transform.translation;
            unit.state.rotation = transform.rotation;

            // Gather environment info
            let environment = gather_environment_info(&unit.state, &all_units_query);

            // Update AI
            if let Some(command) = ai_manager.update_ai_instance(
                unit.state.id,
                &module_name,
                &unit.state,
                &environment,
                time.delta_secs(),
            ) {
                ai_state.current_command = Some(command);
                ai_state.command_start_time = current_time;
                debug!("Unit {} received command: {:?}", unit.state.id, ai_state.current_command);
            }
        }

        ai_state.last_update = current_time;
    }
}

/// System để xử lý di chuyển units
pub fn update_unit_movement(
    mut query: Query<(&mut Transform, &mut UnitComponent, &AiStateComponent)>,
    time: Res<Time>,
) {
    for (mut transform, mut unit, ai_state) in query.iter_mut() {
        if let Some(ref command) = ai_state.current_command {
            match command {
                UnitCommand::MoveTo { target } => {
                    let direction = (*target - transform.translation).normalize();
                    let speed = 5.0; // Base movement speed
                    
                    let movement = direction * speed * time.delta_secs();
                    transform.translation += movement;
                    
                    // Update unit state
                    unit.state.position = transform.translation;
                    unit.state.velocity = direction * speed;
                    unit.state.target_position = Some(*target);
                    unit.state.state = UnitAiState::Moving;
                    
                    // Check if reached target
                    if transform.translation.distance(*target) < 1.0 {
                        unit.state.state = UnitAiState::Idle;
                        unit.state.velocity = Vec3::ZERO;
                        unit.state.target_position = None;
                    }
                }
                UnitCommand::AttackMove { target } => {
                    // Similar to MoveTo but scan for enemies
                    let direction = (*target - transform.translation).normalize();
                    let speed = 3.0; // Slower when attack-moving
                    
                    let movement = direction * speed * time.delta_secs();
                    transform.translation += movement;
                    
                    unit.state.position = transform.translation;
                    unit.state.velocity = direction * speed;
                    unit.state.state = UnitAiState::Moving;
                }
                UnitCommand::Stop => {
                    unit.state.velocity = Vec3::ZERO;
                    unit.state.target_position = None;
                    unit.state.state = UnitAiState::Idle;
                }
                _ => {
                    // Handle other commands
                }
            }
        }
    }
}

/// System để xử lý combat
pub fn update_unit_combat(
    mut query: Query<(&mut UnitComponent, &Transform)>,
    time: Res<Time>,
) {
    let current_time = time.elapsed_secs();
    let mut units_data: Vec<_> = query.iter_mut().collect();
    
    for i in 0..units_data.len() {
        let (unit_a, transform_a) = &units_data[i];
        
        if unit_a.state.state == UnitAiState::Attacking {
            if let Some(target_id) = unit_a.state.target_unit_id {
                // Find target unit
                for j in 0..units_data.len() {
                    if i == j { continue; }
                    
                    let (unit_b, transform_b) = &units_data[j];
                    if unit_b.state.id == target_id && unit_b.state.team_id != unit_a.state.team_id {
                        let distance = transform_a.translation.distance(transform_b.translation);
                        
                        // Check if in attack range (assuming range of 8.0)
                        if distance <= 8.0 {
                            // Perform attack (simple damage calculation)
                            // In real implementation, this would be more sophisticated
                            debug!("Unit {} attacking unit {}", unit_a.state.id, target_id);
                        }
                        break;
                    }
                }
            }
        }
    }
}

/// System để xử lý input từ user
pub fn handle_input(
    mut game_state: ResMut<GameState>,
    keys: Res<ButtonInput<KeyCode>>,
    mut mouse_button_events: EventReader<MouseButtonInput>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    window_query: Query<&Window, With<PrimaryWindow>>,
    unit_query: Query<(Entity, &Transform, &UnitComponent)>,
) {
    // Keyboard input
    if keys.just_pressed(KeyCode::Space) {
        game_state.paused = !game_state.paused;
        info!("Game {}", if game_state.paused { "paused" } else { "resumed" });
    }

    if keys.just_pressed(KeyCode::KeyR) {
        info!("Reloading AI modules...");
        // Hot-reload sẽ được xử lý ở system khác
    }

    if keys.just_pressed(KeyCode::F1) {
        game_state.debug_mode = !game_state.debug_mode;
        info!("Debug mode: {}", game_state.debug_mode);
    }

    // Mouse input for unit selection
    for event in mouse_button_events.read() {
        if event.button == MouseButton::Left && event.state.is_pressed() {
            if let Ok(window) = window_query.single() {
                if let Some(cursor_pos) = window.cursor_position() {
                    if let Ok((camera, camera_transform)) = camera_query.single() {
                        // Ray casting để select units (simplified)
                        info!("Mouse clicked at: {:?}", cursor_pos);
                        // TODO: Implement proper ray casting for unit selection
                    }
                }
            }
        }
    }
}

/// System để hot-reload AI modules
pub fn hot_reload_ai_modules(
    mut ai_manager: ResMut<AiModuleManager>,
) {
    let changed_modules = ai_manager.check_for_changes();
    if !changed_modules.is_empty() {
        info!("Detected changes in modules: {:?}", changed_modules);
        ai_manager.reload_changed_modules(changed_modules);
    }
}

/// Helper function để gather environment info cho AI
fn gather_environment_info(
    unit_state: &UnitState,
    all_units_query: &Query<(&UnitComponent, &Transform)>,
) -> EnvironmentInfo {
    let mut nearby_units = Vec::new();
    let mut nearby_enemies = Vec::new();
    let mut nearby_allies = Vec::new();
    
    let sight_range = 20.0;
    
    for (other_unit, transform) in all_units_query.iter() {
        if other_unit.state.id == unit_state.id {
            continue; // Skip self
        }
        
        let distance = unit_state.position.distance(transform.translation);
        if distance <= sight_range {
            nearby_units.push(other_unit.state.clone());
            
            if other_unit.state.team_id == unit_state.team_id {
                nearby_allies.push(other_unit.state.clone());
            } else {
                nearby_enemies.push(other_unit.state.clone());
            }
        }
    }
    
    EnvironmentInfo {
        nearby_units,
        nearby_enemies,
        nearby_allies,
        terrain_height: 0.0, // TODO: Implement terrain system
        obstacles: Vec::new(), // TODO: Implement obstacle detection
        resources: Vec::new(), // TODO: Implement resource system
    }
}
