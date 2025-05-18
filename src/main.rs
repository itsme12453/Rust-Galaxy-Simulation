use bevy::core_pipeline::bloom::BloomSettings;
use bevy::input::mouse::{ MouseMotion, MouseWheel };
use bevy::math::{ Quat, Vec3 };
use bevy::prelude::*;
use rand::Rng;

const NUM_PARTICLES_PER_GALAXY: usize = 10_000;
const PARTICLE_RADIUS: f32 = 0.01;
const GALAXY_RADIUS: f32 = 5.0;
const GALAXY_SPACING: f32 = 10.0;
const GALAXY_COUNT: usize = 2;
const G: f32 = 30.0;
const DT: f32 = 0.00001;
const PARTICLE_MASS: f32 = 1.0;
const CENTRAL_MASS: f32 = 5e6;
const THETA: f32 = 0.8; // Barnes-Hut opening angle threshold

#[derive(Component)]
struct Velocity(Vec3);
#[derive(Component)]
struct Acceleration(Vec3);
#[derive(Component)]
struct Mass(f32);
#[derive(Component)]
struct ParticleMaterial(Handle<StandardMaterial>);
#[derive(Component)]
struct OrbitCamera {
    radius: f32,
    theta: f32,
    phi: f32,
}
#[derive(Component)]
struct CentralStar;
#[derive(Component)]
struct ExplosionParticle;
#[derive(Component)]
struct ExplosionTimer(Timer);
#[derive(Component)]
struct Shockwave {
    timer: Timer,
    max_radius: f32,
    material: Handle<StandardMaterial>,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::BLACK))
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .add_systems(Update, apply_gravity)
        .add_systems(Update, leapfrog_update)
        // .add_systems(Update, update_particle_colors)
        .add_systems(Update, orbit_camera_control)
        // .add_systems(Update, check_central_star_collision)
        // .add_systems(Update, update_explosions)
        // .add_systems(Update, update_shockwaves)
        .run();
}

fn spawn_n_galaxies(
    n: usize,
    _speed: f32,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    rng: &mut impl Rng
) {
    let circle_radius = (GALAXY_SPACING * (n as f32)) / (2.0 * std::f32::consts::PI);
    let orbital_speed = ((G * CENTRAL_MASS) / (4.0 * circle_radius)).sqrt();

    for i in 0..n {
        let angle = (i as f32) * ((2.0 * std::f32::consts::PI) / (n as f32));
        let x = circle_radius * angle.cos();
        let z = circle_radius * angle.sin();

        let (color, tilt) = if i % 2 == 0 {
            (Color::rgb(1.0, 0.8, 0.2), 30.0)
        } else {
            (Color::rgb(0.4, 0.6, 1.0), -25.0)
        };

        let center_offset = Vec3::new(x, 0.0, z);
        let tangent = Vec3::new(-z, 0.0, x).normalize_or_zero();
        let mut velocity = tangent * orbital_speed;

        velocity = Vec3::new(0.0, 0.0, 0.0);

        spawn_galaxy(
            center_offset,
            color,
            NUM_PARTICLES_PER_GALAXY,
            tilt,
            velocity,
            commands,
            meshes,
            materials,
            rng
        );
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>
) {
    let mut rng = rand::thread_rng();

    let galaxy_speed = 0.0;

    spawn_n_galaxies(
        GALAXY_COUNT,
        galaxy_speed,
        &mut commands,
        &mut meshes,
        &mut materials,
        &mut rng
    );

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(0.0, 0.0, 60.0).looking_at(Vec3::ZERO, Vec3::Y),
            camera: Camera {
                hdr: true,
                ..default()
            },
            ..default()
        },
        BloomSettings {
            intensity: 0.5,
            low_frequency_boost: 2.0,
            low_frequency_boost_curvature: 1.0,
            high_pass_frequency: 0.3,
            prefilter_settings: bevy::core_pipeline::bloom::BloomPrefilterSettings {
                threshold: 0.05,
                threshold_softness: 0.9,
            },
            ..default()
        },
        OrbitCamera {
            radius: 60.0,
            theta: std::f32::consts::FRAC_PI_2,
            phi: std::f32::consts::FRAC_PI_4,
        },
    ));
}

fn spawn_galaxy(
    center: Vec3,
    central_color: Color,
    num: usize,
    tilt_deg: f32,
    center_vel: Vec3,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    rng: &mut impl Rng
) {
    let tilt_rot = Quat::from_rotation_x(tilt_deg.to_radians());

    let sphere_mesh = meshes.add(
        Mesh::from(shape::UVSphere {
            radius: PARTICLE_RADIUS,
            sectors: 8,
            stacks: 4,
        })
    );

    let particle_color = Color::rgb(rng.gen_range(0.0..1.0), 0.0, rng.gen_range(0.0..1.0));
    let base_mat = materials.add(StandardMaterial {
        base_color: particle_color,
        emissive: particle_color * 5.0,
        unlit: true,
        ..default()
    });

    let cen_mesh = meshes.add(
        Mesh::from(shape::UVSphere {
            radius: PARTICLE_RADIUS * 6.0,
            sectors: 16,
            stacks: 8,
        })
    );
    let cen_mat = materials.add(StandardMaterial {
        base_color: central_color,
        emissive: central_color * 500.0,
        unlit: true,
        ..default()
    });

    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1e7,
            color: central_color,
            shadows_enabled: false,
            range: 300.0,
            ..default()
        },
        transform: Transform::from_translation(center),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: cen_mesh.clone(),
            material: cen_mat.clone(),
            transform: Transform::from_translation(center),
            ..default()
        },
        Velocity(center_vel),
        Acceleration(Vec3::ZERO),
        Mass(CENTRAL_MASS),
        CentralStar,
    ));

    for _ in 0..num {
        let ang = rng.gen_range(0.0..std::f32::consts::TAU);
        let r = rng.gen_range(1.0..GALAXY_RADIUS);
        let pos =
            center + tilt_rot * Vec3::new(r * ang.cos(), rng.gen_range(-0.01..0.01), r * ang.sin());

        let speed = ((G * CENTRAL_MASS) / r).sqrt();
        let vel = tilt_rot * Vec3::new(-speed * ang.sin(), 0.0, speed * ang.cos()) + center_vel;

        commands.spawn((
            PbrBundle {
                mesh: sphere_mesh.clone(),
                material: base_mat.clone(),
                transform: Transform::from_translation(pos),
                ..default()
            },
            Velocity(vel),
            Acceleration(Vec3::ZERO),
            Mass(PARTICLE_MASS),
            ParticleMaterial(base_mat.clone()),
        ));
    }
}

const MIN_CELL_SIZE: f32 = PARTICLE_RADIUS * 0.5;

#[derive(Default)]
enum BHNode {
    /// No particles
    #[default]
    Empty,
    /// Single particle
    Leaf {
        center: Vec3,
        mass: f32,
        entity: Entity,
    },
    /// Multiple particles summarized
    Internal {
        center: Vec3,
        mass: f32,
        children: [Box<BHNode>; 8],
    },
}

impl BHNode {
    /// Create an empty internal node
    fn new_internal() -> Self {
        BHNode::Internal {
            center: Vec3::ZERO,
            mass: 0.0,
            children: std::array::from_fn(|_| Box::new(BHNode::Empty)),
        }
    }
}

fn build_bh_tree(
    particles: &[(Entity, Vec3, f32)],
    region_center: Vec3,
    region_half: f32
) -> BHNode {
    let mut root = BHNode::new_internal();
    for &(entity, pos, mass) in particles {
        insert(&mut root, region_center, region_half, entity, pos, mass);
    }
    root
}

fn insert(
    node: &mut BHNode,
    region_center: Vec3,
    region_half: f32,
    entity: Entity,
    pos: Vec3,
    mass: f32
) {
    // If the cell is already too small to subdivide, just accumulate into this node
    if region_half < MIN_CELL_SIZE {
        match node {
            BHNode::Empty => {
                *node = BHNode::Leaf {
                    center: pos,
                    mass,
                    entity,
                };
            }
            | BHNode::Leaf { center: c, mass: m, .. }
            | BHNode::Internal { center: c, mass: m, .. } => {
                // update center of mass
                *c = (*c * *m + pos * mass) / (*m + mass);
                *m += mass;
            }
        }
        return;
    }

    match node {
        BHNode::Empty => {
            *node = BHNode::Leaf {
                center: pos,
                mass,
                entity,
            };
        }
        BHNode::Leaf { center: c, mass: m, entity: e } => {
            // subdivide leaf into internal
            let old_pos = *c;
            let old_mass = *m;
            let old_entity = *e;
            *node = BHNode::new_internal();
            if let BHNode::Internal { center, mass: total_mass, children } = node {
                // update COM for the two particles
                *center = (old_pos * old_mass + pos * mass) / (old_mass + mass);
                *total_mass = old_mass + mass;

                // re-insert both
                for &(ent, p, ma) in &[
                    (old_entity, old_pos, old_mass),
                    (entity, pos, mass),
                ] {
                    let idx = child_index(region_center, p);
                    let offset = child_offset(region_half, idx);
                    insert(
                        &mut children[idx],
                        region_center + offset,
                        region_half * 0.5,
                        ent,
                        p,
                        ma
                    );
                }
            }
        }
        BHNode::Internal { center: com, mass: total_mass, children } => {
            // update COM & mass
            *com = (*com * *total_mass + pos * mass) / (*total_mass + mass);
            *total_mass += mass;
            // descend into the appropriate child
            let idx = child_index(region_center, pos);
            let offset = child_offset(region_half, idx);
            insert(
                &mut children[idx],
                region_center + offset,
                region_half * 0.5,
                entity,
                pos,
                mass
            );
        }
    }
}

/// Octant selection
fn child_index(center: Vec3, pos: Vec3) -> usize {
    ((pos.x > center.x) as usize) |
        (((pos.y > center.y) as usize) << 1) |
        (((pos.z > center.z) as usize) << 2)
}

/// Octant offset
fn child_offset(half: f32, idx: usize) -> Vec3 {
    Vec3::new(
        if (idx & 1) == 0 {
            -half
        } else {
            half
        },
        if (idx & 2) == 0 {
            -half
        } else {
            half
        },
        if (idx & 4) == 0 {
            -half
        } else {
            half
        }
    )
}

/// Compute acceleration from tree
fn compute_acc(node: &BHNode, pos: Vec3, region_center: Vec3, region_half: f32) -> Vec3 {
    match node {
        BHNode::Empty => Vec3::ZERO,
        BHNode::Leaf { center: c, mass: m, .. } => {
            let d = *c - pos;
            if d.length_squared() < 0.0001 {
                return Vec3::ZERO;
            }
            (G * m * d.normalize()) / d.length_squared().max(0.01)
        }
        BHNode::Internal { center: com, mass: m, children } => {
            let d = *com - pos;
            let size = region_half * 2.0;
            if size / d.length() < THETA {
                (G * *m * d.normalize()) / d.length_squared().max(0.01)
            } else {
                let mut acc = Vec3::ZERO;
                for (i, child) in children.iter().enumerate() {
                    if let BHNode::Empty = **child {
                        continue;
                    }
                    let offset = child_offset(region_half, i);
                    acc += compute_acc(child, pos, region_center + offset, region_half * 0.5);
                }
                acc
            }
        }
    }
}

fn apply_gravity(mut q: Query<(Entity, &Transform, &Mass, &mut Acceleration)>) {
    let data: Vec<_> = q
        .iter()
        .map(|(e, t, m, _)| (e, t.translation, m.0))
        .collect();
    let region_center = Vec3::ZERO;
    let region_half = GALAXY_SPACING * 10.0;
    let tree = build_bh_tree(&data, region_center, region_half);
    for (_e, t, _m, mut a) in q.iter_mut() {
        a.0 = compute_acc(&tree, t.translation, region_center, region_half);
    }
}

fn leapfrog_update(mut q: Query<(&mut Transform, &mut Velocity, &Acceleration)>) {
    for (mut t, mut v, a) in q.iter_mut() {
        v.0 += 0.5 * a.0 * DT;
        t.translation += v.0 * DT;
        v.0 += 0.5 * a.0 * DT;
    }
}

fn update_particle_colors(
    mut mats: ResMut<Assets<StandardMaterial>>,
    q: Query<(&Acceleration, &ParticleMaterial)>
) {
    if let Some((a, ParticleMaterial(h))) = q.iter().next() {
        if let Some(m) = mats.get_mut(h) {
            let t = a.0.length().clamp(0.0, 1.0);
            let c = Color::rgb(t, 0.0, 1.0 - t);
            m.base_color = c;
            m.emissive = c * 5.0;
        }
    }
}

fn orbit_camera_control(
    mut mm: EventReader<MouseMotion>,
    mut sw: EventReader<MouseWheel>,
    mb: Res<Input<MouseButton>>,
    mut q: Query<(&mut OrbitCamera, &mut Transform)>
) {
    let mut d = Vec2::ZERO;
    for e in mm.read() {
        d += e.delta;
    }
    let s: f32 = sw
        .read()
        .map(|e| e.y)
        .sum();
    for (mut o, mut t) in q.iter_mut() {
        if mb.pressed(MouseButton::Left) {
            o.theta -= d.x * 0.01;
            o.phi = (o.phi - d.y * 0.01).clamp(0.05, std::f32::consts::PI - 0.05);
        }
        o.radius *= (0.95_f32).powf(s);
        let x = o.radius * o.phi.sin() * o.theta.cos();
        let y = o.radius * o.phi.cos();
        let z = o.radius * o.phi.sin() * o.theta.sin();
        t.translation = Vec3::new(x, y, z);
        t.look_at(Vec3::ZERO, Vec3::Y);
    }
}

fn check_central_star_collision(
    mut commands: Commands,
    star_q: Query<(Entity, &Transform, &Velocity), With<CentralStar>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>
) {
    let stars: Vec<_> = star_q.iter().collect();
    if stars.len() < 2 {
        return;
    }
    let (e1, t1, v1) = stars[0];
    let (e2, t2, v2) = stars[1];
    if t1.translation.distance(t2.translation) < 0.5 {
        let center = (t1.translation + t2.translation) / 2.0;
        let combined_vel = (v1.0 + v2.0) / 2.0;
        commands.entity(e1).despawn_recursive();
        commands.entity(e2).despawn_recursive();

        let ring_mesh = meshes.add(
            Mesh::from(shape::UVSphere {
                radius: 1.0,
                sectors: 16,
                stacks: 16,
            })
        );
        let ring_mat = mats.add(StandardMaterial {
            base_color: Color::rgba(1.0, 1.0, 1.0, 1.0),
            emissive: Color::WHITE * 300.0,
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: ring_mesh.clone(),
                material: ring_mat.clone(),
                transform: Transform {
                    translation: center,
                    scale: Vec3::splat(0.1),
                    ..default()
                },
                ..default()
            },
            Shockwave {
                timer: Timer::from_seconds(1.5, TimerMode::Once),
                max_radius: 15.0,
                material: ring_mat.clone(),
            },
        ));
        
        let fusion_mesh = meshes.add(
            Mesh::from(shape::UVSphere {
                radius: 0.15,
                sectors: 16,
                stacks: 8,
            })
        );
        let fusion_mat = mats.add(StandardMaterial {
            base_color: Color::rgb(1.0, 0.3, 0.6),
            emissive: Color::rgb(1.0, 0.3, 0.6) * 200.0,
            unlit: true,
            ..default()
        });
        commands.spawn((
            PbrBundle {
                mesh: fusion_mesh.clone(),
                material: fusion_mat.clone(),
                transform: Transform::from_translation(center),
                ..default()
            },
            Velocity(combined_vel),
            Acceleration(Vec3::ZERO),
            Mass(CENTRAL_MASS * 2.0),
            CentralStar,
            Name::new("FusionStar"),
        ));
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: 200_000.0,
                range: 300.0,
                shadows_enabled: true,
                color: Color::rgb(1.0, 0.2, 0.5),
                ..default()
            },
            transform: Transform::from_translation(center),
            ..default()
        });
    }
}

fn update_explosions(
    time: Res<Time>,
    mut cmds: Commands,
    mut q: Query<(Entity, &mut Transform, &Velocity, &mut ExplosionTimer), With<ExplosionParticle>>
) {
    for (e, mut t, v, mut timer) in q.iter_mut() {
        t.translation += v.0 * time.delta_seconds();
        timer.0.tick(time.delta());
        if timer.0.finished() {
            cmds.entity(e).despawn_recursive();
        }
    }
}

fn update_shockwaves(
    time: Res<Time>,
    mut cmds: Commands,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut q: Query<(Entity, &mut Transform, &mut Shockwave)>
) {
    for (e, mut t, mut s) in q.iter_mut() {
        s.timer.tick(time.delta());
        let progress = s.timer.elapsed_secs() / s.timer.duration().as_secs_f32();
        t.scale = Vec3::splat(s.max_radius * progress);
        if let Some(mat) = mats.get_mut(&s.material) {
            mat.base_color.set_a(1.0 - progress);
        }
        if s.timer.finished() {
            cmds.entity(e).despawn_recursive();
        }
    }
}
