use bevy::prelude::*;
use shared::*;

/// Component chính cho mỗi unit trong game
#[derive(Component)]
pub struct UnitComponent {
    pub state: UnitState,
    pub ai_module_name: Option<String>,
}

/// Component cho việc di chuyển
#[derive(Component)]
pub struct MovementComponent {
    pub target_position: Option<Vec3>,
    pub speed: f32,
    pub path: Vec<Vec3>,
}

/// Component cho combat
#[derive(Component)]
pub struct CombatComponent {
    pub stats: CombatStats,
    pub last_attack_time: f32,
    pub current_target: Option<Entity>,
}

/// Component để đánh dấu unit đã chết
#[derive(Component)]
pub struct DeadUnit;

/// Component cho AI state riêng của mỗi unit
#[derive(Component)]
pub struct AiStateComponent {
    pub last_update: f32,
    pub update_interval: f32,
    pub current_command: Option<UnitCommand>,
    pub command_start_time: f32,
}

impl Default for MovementComponent {
    fn default() -> Self {
        Self {
            target_position: None,
            speed: 5.0,
            path: Vec::new(),
        }
    }
}

impl Default for CombatComponent {
    fn default() -> Self {
        Self {
            stats: CombatStats {
                damage: 25.0,
                armor: 5.0,
                attack_range: 10.0,
                attack_speed: 1.0,
                movement_speed: 5.0,
                sight_range: 15.0,
            },
            last_attack_time: 0.0,
            current_target: None,
        }
    }
}

impl Default for AiStateComponent {
    fn default() -> Self {
        Self {
            last_update: 0.0,
            update_interval: 0.1, // Update AI mỗi 100ms
            current_command: None,
            command_start_time: 0.0,
        }
    }
}
