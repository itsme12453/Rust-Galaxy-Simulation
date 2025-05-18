use bevy::core_pipeline::bloom::BloomSettings;
use bevy::input::mouse::{ MouseMotion, MouseWheel };
use bevy::math::{ Quat, Vec3 };
use bevy::prelude::*;
use rand::Rng;
use rayon::prelude::*;

const NUM_PARTICLES_PER_GALAXY: usize = 70_000;
const PARTICLE_RADIUS: f32 = 0.01;
const GALAXY_RADIUS: f32 = 5.0;
const GALAXY_SPACING: f32 = 10.0;
const GALAXY_COUNT: usize = 1;
const G: f32 = 1.0;
const DT: f32 = 0.0001;
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

    // let rand_mass: f32 = rng.gen_range(15e6..30e6);

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

struct BHNodeArena {
    nodes: Vec<BHNode>,
}

impl BHNodeArena {
    fn new() -> Self {
        Self { nodes: Vec::with_capacity(1024) }
    }
    fn alloc(&mut self, node: BHNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }
    fn get(&self, idx: usize) -> &BHNode {
        &self.nodes[idx]
    }
    fn get_mut(&mut self, idx: usize) -> &mut BHNode {
        &mut self.nodes[idx]
    }
}

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
        children: [Option<usize>; 8],
    },
}

impl BHNode {
    /// Create an empty internal node
    fn new_internal() -> Self {
        BHNode::Internal {
            center: Vec3::ZERO,
            mass: 0.0,
            children: [None; 8],
        }
    }
}

fn build_bh_tree(
    particles: &[(Entity, Vec3, f32)],
    region_center: Vec3,
    region_half: f32,
    arena: &mut BHNodeArena
) -> usize {
    let root_idx = arena.alloc(BHNode::new_internal());
    for &(entity, pos, mass) in particles {
        insert(root_idx, region_center, region_half, entity, pos, mass, arena);
    }
    root_idx
}

fn partition_particles(
    particles: &[(Entity, Vec3, f32)],
    region_center: Vec3,
) -> [Vec<(Entity, Vec3, f32)>; 8] {
    let mut buckets: [Vec<(Entity, Vec3, f32)>; 8] = Default::default();
    for &(e, p, m) in particles {
        let idx = child_index(region_center, p);
        buckets[idx].push((e, p, m));
    }
    buckets
}

fn build_bh_tree_parallel(
    particles: &[(Entity, Vec3, f32)],
    region_center: Vec3,
    region_half: f32,
    arena: &mut BHNodeArena,
) -> usize {
    if particles.len() <= 1000 || region_half < MIN_CELL_SIZE {
        // Fallback to sequential for small sets or tiny regions
        return build_bh_tree(particles, region_center, region_half, arena);
    }
    let buckets = partition_particles(particles, region_center);
    // Each thread gets its own arena
    let subtrees: Vec<(Vec<BHNode>, Option<usize>)> = buckets
        .into_par_iter()
        .enumerate()
        .map(|(i, bucket)| {
            if bucket.is_empty() {
                (Vec::new(), None)
            } else {
                let mut local_arena = BHNodeArena { nodes: Vec::new() };
                let offset = child_offset(region_half, i);
                let idx = build_bh_tree(&bucket, region_center + offset, region_half * 0.5, &mut local_arena);
                (local_arena.nodes, Some(idx))
            }
        })
        .collect();
    // Merge all sub-arenas into the main arena, updating indices
    let mut children_idxs = [None; 8];
    let mut node_offset = arena.nodes.len();
    let mut idx_map: Vec<Vec<usize>> = Vec::with_capacity(8);
    for (nodes, idx_opt) in &subtrees {
        let mut map = Vec::with_capacity(nodes.len());
        for _ in 0..nodes.len() {
            map.push(arena.nodes.len());
            arena.nodes.push(BHNode::Empty); // placeholder
        }
        idx_map.push(map);
    }
    // Copy nodes and update child indices
    for (octant, (nodes, idx_opt)) in subtrees.iter().enumerate() {
        let map = &idx_map[octant];
        for (old_idx, node) in nodes.iter().enumerate() {
            let new_idx = map[old_idx];
            let new_node = match node {
                BHNode::Internal { center, mass, children } => {
                    let mut new_children = [None; 8];
                    for (i, child) in children.iter().enumerate() {
                        if let Some(child_idx) = child {
                            new_children[i] = Some(map[*child_idx]);
                        }
                    }
                    BHNode::Internal {
                        center: *center,
                        mass: *mass,
                        children: new_children,
                    }
                }
                BHNode::Leaf { center, mass, entity } => {
                    BHNode::Leaf { center: *center, mass: *mass, entity: *entity }
                }
                BHNode::Empty => BHNode::Empty,
            };
            arena.nodes[new_idx] = new_node;
        }
        if let Some(idx) = idx_opt {
            children_idxs[octant] = Some(map[*idx]);
        }
    }
    // Compute center of mass and total mass
    let mut center = Vec3::ZERO;
    let mut mass = 0.0;
    for &child_idx in &children_idxs {
        if let Some(idx) = child_idx {
            match &arena.nodes[idx] {
                BHNode::Internal { center: c, mass: m, .. }
                | BHNode::Leaf { center: c, mass: m, .. } => {
                    center += *c * *m;
                    mass += *m;
                }
                _ => {}
            }
        }
    }
    if mass > 0.0 {
        center /= mass;
    }
    arena.alloc(BHNode::Internal {
        center,
        mass,
        children: children_idxs,
    })
}

fn insert(
    node_idx: usize,
    region_center: Vec3,
    region_half: f32,
    entity: Entity,
    pos: Vec3,
    mass: f32,
    arena: &mut BHNodeArena
) {
    let need_accumulate = region_half < MIN_CELL_SIZE;
    if need_accumulate {
        {
            let node = arena.get_mut(node_idx);
            match node {
                BHNode::Empty => {
                    *node = BHNode::Leaf { center: pos, mass, entity };
                }
                | BHNode::Leaf { center: c, mass: m, .. }
                | BHNode::Internal { center: c, mass: m, .. } => {
                    *c = (*c * *m + pos * mass) / (*m + mass);
                    *m += mass;
                }
            }
        }
        return;
    }

    // Determine node type, then drop the mutable borrow
    let node_type = {
        let node = arena.get_mut(node_idx);
        match node {
            BHNode::Empty => 0,
            BHNode::Leaf { .. } => 1,
            BHNode::Internal { .. } => 2,
        }
    };

    match node_type {
        0 => {
            let node = arena.get_mut(node_idx);
            *node = BHNode::Leaf { center: pos, mass, entity };
        }
        1 => {
            // Extract old values, then drop the borrow
            let (old_pos, old_mass, old_entity) = {
                let node = arena.get_mut(node_idx);
                if let BHNode::Leaf { center, mass, entity } = node {
                    (*center, *mass, *entity)
                } else {
                    unreachable!()
                }
            };
            // Replace with internal
            {
                let node = arena.get_mut(node_idx);
                *node = BHNode::new_internal();
            }
            // Update COM and mass
            {
                let node = arena.get_mut(node_idx);
                if let BHNode::Internal { center, mass: m, .. } = node {
                    *center = (old_pos * old_mass + pos * mass) / (old_mass + mass);
                    *m = old_mass + mass;
                }
            }
            // Insert both particles
            for &(ent, p, ma) in &[(old_entity, old_pos, old_mass), (entity, pos, mass)] {
                let idx = child_index(region_center, p);
                let offset = child_offset(region_half, idx);
                
                let child_idx = {
                    let mut need_alloc = false;
                    {
                        let node = arena.get_mut(node_idx);
                        if let BHNode::Internal { children, .. } = node {
                            if children[idx].is_none() {
                                need_alloc = true;
                            }
                        }
                    }
                    if need_alloc {
                        let new_idx = arena.alloc(BHNode::Empty);
                        let node = arena.get_mut(node_idx);
                        if let BHNode::Internal { children, .. } = node {
                            children[idx] = Some(new_idx);
                        }
                    }
                    let node = arena.get_mut(node_idx);
                    if let BHNode::Internal { children, .. } = node {
                        children[idx].unwrap()
                    } else {
                        unreachable!()
                    }
                };
                insert(child_idx, region_center + offset, region_half * 0.5, ent, p, ma, arena);
            }
        }
        2 => {
            // Update COM and mass
            {
                let node = arena.get_mut(node_idx);
                if let BHNode::Internal { center, mass: m, .. } = node {
                    *center = (*center * *m + pos * mass) / (*m + mass);
                    *m += mass;
                }
            }
            let idx = child_index(region_center, pos);
            let offset = child_offset(region_half, idx);
            
            let child_idx = {
                let mut need_alloc = false;
                {
                    let node = arena.get_mut(node_idx);
                    if let BHNode::Internal { children, .. } = node {
                        if children[idx].is_none() {
                            need_alloc = true;
                        }
                    }
                }
                if need_alloc {
                    let new_idx = arena.alloc(BHNode::Empty);
                    let node = arena.get_mut(node_idx);
                    if let BHNode::Internal { children, .. } = node {
                        children[idx] = Some(new_idx);
                    }
                }
                let node = arena.get_mut(node_idx);
                if let BHNode::Internal { children, .. } = node {
                    children[idx].unwrap()
                } else {
                    unreachable!()
                }
            };
            insert(child_idx, region_center + offset, region_half * 0.5, entity, pos, mass, arena);
        }
        _ => unreachable!(),
    }
}

fn child_index(center: Vec3, pos: Vec3) -> usize {
    ((pos.x > center.x) as usize) |
        (((pos.y > center.y) as usize) << 1) |
        (((pos.z > center.z) as usize) << 2)
}

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

fn compute_acc(arena: &BHNodeArena, node_idx: usize, pos: Vec3, region_center: Vec3, region_half: f32) -> Vec3 {
    match arena.get(node_idx) {
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
                    if let Some(child_idx) = child {
                        let offset = child_offset(region_half, i);
                        acc += compute_acc(arena, *child_idx, pos, region_center + offset, region_half * 0.5);
                    }
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
    let mut arena = BHNodeArena::new();
    let root_idx = if data.len() > 2000 {
        build_bh_tree_parallel(&data, region_center, region_half, &mut arena)
    } else {
        build_bh_tree(&data, region_center, region_half, &mut arena)
    };
    q.par_iter_mut().for_each(|(_e, t, _m, mut a)| {
        a.0 = compute_acc(&arena, root_idx, t.translation, region_center, region_half);
    });
    // for (_e, t, _m, mut a) in q.iter_mut() {
    //     a.0 = compute_acc(&tree, t.translation, region_center, region_half);
    // }
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
