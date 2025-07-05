use bevy::prelude::*;
use shared::*;

/// Utilities cho game engine

/// Convert Bevy Vec3 to game world coordinates
pub fn bevy_to_world_pos(bevy_pos: Vec3) -> Vec3 {
    bevy_pos
}

/// Convert game world coordinates to Bevy Vec3
pub fn world_to_bevy_pos(world_pos: Vec3) -> Vec3 {
    world_pos
}

/// Calculate distance between two units
pub fn unit_distance(unit_a: &UnitState, unit_b: &UnitState) -> f32 {
    unit_a.position.distance(unit_b.position)
}

/// Check if unit is in range of target
pub fn is_in_range(unit: &UnitState, target: &UnitState, range: f32) -> bool {
    unit_distance(unit, target) <= range
}

/// Get direction vector from unit to target
pub fn direction_to_target(unit: &UnitState, target_pos: Vec3) -> Vec3 {
    (target_pos - unit.position).normalize()
}

/// Apply damage to unit with armor calculation
pub fn apply_damage(unit: &mut UnitState, damage: f32, armor: f32) {
    let effective_damage = (damage - armor).max(0.0);
    unit.health = (unit.health - effective_damage).max(0.0);
}

/// Check if unit is alive
pub fn is_unit_alive(unit: &UnitState) -> bool {
    unit.health > 0.0
}

/// Get health percentage
pub fn health_percentage(unit: &UnitState) -> f32 {
    if unit.max_health > 0.0 {
        unit.health / unit.max_health
    } else {
        0.0
    }
}

/// Linear interpolation for smooth movement
pub fn lerp_vec3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// Simple pathfinding (placeholder for A* implementation)
pub fn find_path(start: Vec3, end: Vec3, _obstacles: &[Vec3]) -> Vec<Vec3> {
    // Simplified: direct path for now
    // TODO: Implement proper A* pathfinding
    vec![start, end]
}

/// Get random position within radius
pub fn random_position_in_radius(center: Vec3, radius: f32) -> Vec3 {
    use std::f32::consts::TAU;
    
    let angle = fastrand::f32() * TAU;
    let distance = fastrand::f32() * radius;
    
    Vec3::new(
        center.x + angle.cos() * distance,
        center.y,
        center.z + angle.sin() * distance,
    )
}

/// Format time for display
pub fn format_time(seconds: f32) -> String {
    let minutes = (seconds / 60.0) as u32;
    let seconds = (seconds % 60.0) as u32;
    format!("{:02}:{:02}", minutes, seconds)
}

/// Debug helpers
pub mod debug {
    use bevy::prelude::*;
    use super::*;

    /// Draw debug line between two points
    pub fn draw_debug_line(
        start: Vec3,
        end: Vec3,
        color: Color,
        gizmos: &mut Gizmos,
    ) {
        gizmos.line(start, end, color);
    }

    /// Draw debug circle at position
    pub fn draw_debug_circle(
        position: Vec3,
        radius: f32,
        color: Color,
        gizmos: &mut Gizmos,
    ) {
        gizmos.circle(position, radius, color);
    }

    /// Print unit debug info
    pub fn print_unit_debug(unit: &UnitState) {
        println!(
            "Unit {}: {:?} at {:?}, HP: {:.1}/{:.1}, State: {:?}",
            unit.id,
            unit.unit_type,
            unit.position,
            unit.health,
            unit.max_health,
            unit.state
        );
    }
}
