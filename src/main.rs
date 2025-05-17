use bevy::prelude::*;
use bevy::input::mouse::{MouseMotion, MouseWheel};
use rand::Rng;

const NUM_PARTICLES: usize = 1000;
const PARTICLE_RADIUS: f32 = 0.01;
const CIRCLE_RADIUS: f32 = 5.0;
const G: f32 = 1.0;
const DT: f32 = 0.001;
const PARTICLE_MASS: f32 = 1.0;

#[derive(Component)]
struct Velocity(Vec3);

#[derive(Component)]
struct Acceleration(Vec3);

#[derive(Component)]
struct Mass(f32);

#[derive(Component)]
struct OrbitCamera {
    radius: f32,
    theta: f32,
    phi: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, apply_gravity)
        .add_systems(Update, leapfrog_update)
        .add_systems(Update, orbit_camera_control)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        ..default()
    });

    let mesh = meshes.add(Mesh::from(shape::UVSphere {
        radius: PARTICLE_RADIUS,
        sectors: 8,
        stacks: 4,
    }));

    let mut rng = rand::thread_rng();

    for _ in 0..NUM_PARTICLES {
        let angle = rng.gen_range(0.0..std::f32::consts::TAU);
        let radius = rng.gen_range(0.5..CIRCLE_RADIUS);
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        let y = rng.gen_range(-0.1..0.1);

        commands.spawn((
            PbrBundle {
                mesh: mesh.clone(),
                material: material.clone(),
                transform: Transform::from_translation(Vec3::new(x, y, z)),
                ..default()
            },
            Velocity(Vec3::ZERO),
            Acceleration(Vec3::ZERO),
            Mass(PARTICLE_MASS),
        ));
    }

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
            ..default()
        },
        OrbitCamera {
            radius: 10.0,
            theta: std::f32::consts::FRAC_PI_2, // facing center from side
            phi: std::f32::consts::FRAC_PI_4,   // tilt downward
        },
    ));
}

fn apply_gravity(
    mut query: Query<(&Transform, &Mass, &mut Acceleration)>,
) {
    let entities: Vec<_> = query.iter_mut().collect();
    let positions: Vec<Vec3> = entities.iter().map(|(t, _, _)| t.translation).collect();
    let masses: Vec<f32> = entities.iter().map(|(_, m, _)| m.0).collect();

    for (i, (_, _, mut acc)) in entities.into_iter().enumerate() {
        let pos_i = positions[i];
        let mut force = Vec3::ZERO;

        for (j, &pos_j) in positions.iter().enumerate() {
            if i == j {
                continue;
            }

            let dir = pos_j - pos_i;
            let dist_sq = dir.length_squared() + 0.01;
            force += G * masses[j] * dir.normalize() / dist_sq;
        }

        acc.0 = force;
    }
}

fn leapfrog_update(
    mut query: Query<(&mut Transform, &mut Velocity, &Acceleration)>,
) {
    for (mut transform, mut velocity, acc) in query.iter_mut() {
        velocity.0 += 0.5 * acc.0 * DT;
        transform.translation += velocity.0 * DT;
        velocity.0 += 0.5 * acc.0 * DT;
    }
}

fn orbit_camera_control(
    time: Res<Time>,
    mut mouse_motion_events: EventReader<MouseMotion>,
    mut scroll_evr: EventReader<MouseWheel>,
    mouse_buttons: Res<Input<MouseButton>>,
    mut query: Query<(&mut OrbitCamera, &mut Transform)>,
) {
    let mut delta = Vec2::ZERO;
    for event in mouse_motion_events.read() {
        delta += event.delta;
    }

    let scroll: f32 = scroll_evr.read().map(|e| e.y).sum();

    for (mut orbit, mut transform) in query.iter_mut() {
        if mouse_buttons.pressed(MouseButton::Left) {
            orbit.theta -= delta.x * 0.01;
            orbit.phi = (orbit.phi - delta.y * 0.01).clamp(0.05, std::f32::consts::PI - 0.05);
        }

        orbit.radius *= 0.95f32.powf(scroll);

        // Convert spherical coordinates to Cartesian
        let x = orbit.radius * orbit.phi.sin() * orbit.theta.cos();
        let y = orbit.radius * orbit.phi.cos();
        let z = orbit.radius * orbit.phi.sin() * orbit.theta.sin();

        transform.translation = Vec3::new(x, y, z);
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
