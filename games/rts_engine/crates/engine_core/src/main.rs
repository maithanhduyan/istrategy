mod ai_loader;
mod components;
mod systems;
mod ui;
mod utils;

use ai_loader::AiModuleManager;
use bevy::prelude::*;
use bevy::window::WindowResolution;
use components::*;
use shared::*;
use systems::*;
use ui::*;

fn main() {
    env_logger::init();

    App::new()
        .add_plugins((DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "RTS Engine - Real-Time Strategy".into(),
                resolution: WindowResolution::new(1280.0, 720.0),
                ..default()
            }),
            ..default()
        }),))
        .init_resource::<AiModuleManager>()
        .init_resource::<GameState>()
        .add_systems(
            Startup,
            (setup_camera, setup_ground, setup_ui, spawn_test_units),
        )
        .add_systems(
            Update,
            (
                update_ai_modules,
                update_unit_ai,
                update_unit_movement,
                update_unit_combat,
                handle_input,
                update_ui,
            ),
        )
        .add_systems(Update, hot_reload_ai_modules)
        .run();
}

/// Resource để quản lý trạng thái game
#[derive(Resource, Default)]
pub struct GameState {
    pub paused: bool,
    pub time_scale: f32,
    pub selected_units: Vec<Entity>,
    pub camera_speed: f32,
    pub debug_mode: bool,
}

impl GameState {
    pub fn new() -> Self {
        Self {
            paused: false,
            time_scale: 1.0,
            selected_units: Vec::new(),
            camera_speed: 500.0,
            debug_mode: false,
        }
    }
}

/// Setup camera 3D
fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 50.0, 50.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// Setup ground plane
fn setup_ground(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Ground plane
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(100.0, 100.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.5, 0.3),
            ..default()
        })),
        Transform::default(),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 32000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::ZYX,
            0.0,
            1.0,
            -std::f32::consts::FRAC_PI_4,
        )),
    ));
}

/// Spawn một số unit test
fn spawn_test_units(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Tank unit
    let tank_mesh = meshes.add(Cuboid::new(2.0, 1.0, 3.0));
    let tank_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.8, 0.2),
        ..default()
    });

    commands.spawn((
        Mesh3d(tank_mesh.clone()),
        MeshMaterial3d(tank_material.clone()),
        Transform::from_xyz(-10.0, 0.5, 0.0),
        UnitComponent {
            state: UnitState {
                id: 1,
                position: Vec3::new(-10.0, 0.5, 0.0),
                unit_type: UnitType::Tank,
                team_id: 1,
                ..default()
            },
            ai_module_name: Some("tank_ai".to_string()),
        },
        Name::new("Tank Unit"),
    ));

    // Scout unit
    let scout_mesh = meshes.add(Sphere::new(0.8));
    let scout_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.2, 0.2),
        ..default()
    });

    commands.spawn((
        Mesh3d(scout_mesh),
        MeshMaterial3d(scout_material),
        Transform::from_xyz(10.0, 0.4, 0.0),
        UnitComponent {
            state: UnitState {
                id: 2,
                position: Vec3::new(10.0, 0.4, 0.0),
                unit_type: UnitType::Scout,
                team_id: 1,
                ..default()
            },
            ai_module_name: Some("scout_ai".to_string()),
        },
        Name::new("Scout Unit"),
    ));

    // Medic unit
    let medic_mesh = meshes.add(Cylinder::new(0.6, 1.5));
    let medic_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.2, 0.8),
        ..default()
    });

    commands.spawn((
        Mesh3d(medic_mesh),
        MeshMaterial3d(medic_material),
        Transform::from_xyz(0.0, 0.75, 10.0),
        UnitComponent {
            state: UnitState {
                id: 3,
                position: Vec3::new(0.0, 0.75, 10.0),
                unit_type: UnitType::Medic,
                team_id: 1,
                ..default()
            },
            ai_module_name: Some("medic_ai".to_string()),
        },
        Name::new("Medic Unit"),
    ));

    info!("Spawned 3 test units: Tank, Scout, Medic");
}
