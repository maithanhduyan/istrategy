use crate::ai_loader::AiModuleManager;
use crate::GameState;
use bevy::prelude::*;

/// System để setup UI
pub fn setup_ui(mut commands: Commands) {
    // Root UI node
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                ..default()
            },
            BackgroundColor(Color::NONE),
        ))
        .with_children(|parent| {
            // Top panel - Game info
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(50.0),
                        padding: UiRect::all(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.1, 0.1, 0.1, 0.8)),
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text::new(
                            "RTS Engine - Press SPACE to pause, R to reload AI, F1 for debug",
                        ),
                        TextFont {
                            font_size: 16.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });

            // Bottom panel - Unit info and controls
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(100.0),
                        align_self: AlignSelf::FlexEnd,
                        padding: UiRect::all(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.1, 0.1, 0.1, 0.8)),
                ))
                .with_children(|parent| {
                    parent.spawn((
                        Text::new("No unit selected"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                        UnitInfoText,
                    ));
                });
        });
}

/// Marker component for unit info text
#[derive(Component)]
pub struct UnitInfoText;

/// System để update UI
pub fn update_ui(
    game_state: Res<GameState>,
    ai_manager: Res<AiModuleManager>,
    mut ui_query: Query<&mut Text, With<UnitInfoText>>,
    unit_query: Query<&crate::components::UnitComponent>,
) {
    for mut text in ui_query.iter_mut() {
        let loaded_modules = ai_manager.get_loaded_modules();
        let unit_count = unit_query.iter().count();

        **text = format!(
            "Units: {} | Loaded AI Modules: {} | Game: {} | Debug: {}",
            unit_count,
            loaded_modules.len(),
            if game_state.paused {
                "PAUSED"
            } else {
                "RUNNING"
            },
            if game_state.debug_mode { "ON" } else { "OFF" }
        );
    }
}
