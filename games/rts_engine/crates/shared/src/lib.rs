use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export Vec3 and other common types for AI modules
pub use bevy::math::Quat;
pub use bevy::math::Vec3;

/// Trạng thái của một đơn vị trong game
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitState {
    pub id: u32,
    pub position: Vec3,
    pub rotation: Quat,
    pub health: f32,
    pub max_health: f32,
    pub unit_type: UnitType,
    pub team_id: u32,
    pub velocity: Vec3,
    pub target_position: Option<Vec3>,
    pub target_unit_id: Option<u32>,
    pub energy: f32,
    pub max_energy: f32,
    pub state: UnitAiState,
}

/// Loại đơn vị
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitType {
    Tank,
    Scout,
    Medic,
    Worker,
    Commander,
}

/// Trạng thái AI của đơn vị
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitAiState {
    Idle,
    Moving,
    Attacking,
    Patrolling,
    Retreating,
    Healing,
    Gathering,
    Building,
}

/// Lệnh mà AI có thể đưa ra cho đơn vị
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnitCommand {
    MoveTo { target: Vec3 },
    AttackUnit { target_id: u32 },
    AttackMove { target: Vec3 },
    Patrol { waypoints: Vec<Vec3> },
    Stop,
    Retreat { safe_position: Vec3 },
    Heal { target_id: u32 },
    Follow { target_id: u32 },
    Guard { position: Vec3, radius: f32 },
}

/// Thông tin về môi trường xung quanh đơn vị
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub nearby_units: Vec<UnitState>,
    pub nearby_enemies: Vec<UnitState>,
    pub nearby_allies: Vec<UnitState>,
    pub terrain_height: f32,
    pub obstacles: Vec<Vec3>,
    pub resources: Vec<ResourceNode>,
}

/// Node tài nguyên trong game
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceNode {
    pub id: u32,
    pub position: Vec3,
    pub resource_type: ResourceType,
    pub amount: f32,
    pub max_amount: f32,
}

/// Loại tài nguyên
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    Metal,
    Energy,
    Food,
    Rare,
}

/// Interface chính cho AI Module
/// Mỗi AI module phải implement trait này
pub trait UnitAiModule: Send + Sync {
    /// Tên của AI module
    fn name(&self) -> &'static str;

    /// Version của AI module
    fn version(&self) -> &'static str;

    /// Loại đơn vị mà AI này hỗ trợ
    fn supported_unit_types(&self) -> Vec<UnitType>;

    /// Update logic AI cho đơn vị
    /// Trả về lệnh mà đơn vị nên thực hiện
    fn update(
        &mut self,
        unit_state: &UnitState,
        environment: &EnvironmentInfo,
        delta_time: f32,
    ) -> Option<UnitCommand>;

    /// Khởi tạo AI cho đơn vị mới
    fn initialize(&mut self, unit_state: &UnitState);

    /// Cleanup khi đơn vị bị destroy
    fn cleanup(&mut self, unit_id: u32);
}

/// Metadata của AI Module để engine có thể load
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiModuleMetadata {
    pub name: String,
    pub version: String,
    pub supported_unit_types: Vec<UnitType>,
    pub description: String,
    pub author: String,
}

/// Function signature để tạo AI module từ .so file
pub type CreateAiModuleFn = unsafe extern "C" fn() -> *mut dyn UnitAiModule;

/// Function signature để lấy metadata từ .so file
pub type GetModuleMetadataFn = unsafe extern "C" fn() -> AiModuleMetadata;

/// Trait để serialize/deserialize AI state (cho hot-reload)
pub trait AiStateSerializable {
    fn serialize_state(&self) -> Vec<u8>;
    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), String>;
}

/// Combat stats cho đơn vị
#[repr(C)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombatStats {
    pub damage: f32,
    pub armor: f32,
    pub attack_range: f32,
    pub attack_speed: f32,
    pub movement_speed: f32,
    pub sight_range: f32,
}

impl Default for UnitState {
    fn default() -> Self {
        Self {
            id: 0,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            health: 100.0,
            max_health: 100.0,
            unit_type: UnitType::Tank,
            team_id: 0,
            velocity: Vec3::ZERO,
            target_position: None,
            target_unit_id: None,
            energy: 100.0,
            max_energy: 100.0,
            state: UnitAiState::Idle,
        }
    }
}

impl Default for EnvironmentInfo {
    fn default() -> Self {
        Self {
            nearby_units: Vec::new(),
            nearby_enemies: Vec::new(),
            nearby_allies: Vec::new(),
            terrain_height: 0.0,
            obstacles: Vec::new(),
            resources: Vec::new(),
        }
    }
}
