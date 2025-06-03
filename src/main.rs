use {
    ::catppuccin::{
        Color as CColor,
        FlavorColors,
        FlavorName,
        PALETTE,
    },
    ::clap::Parser,
    ::color_eyre::{
        eyre::{
            Context as _,
            OptionExt as _,
            Result,
        },
        install as install_eyre,
    },
    ::futures::executor::block_on,
    ::nalgebra::{
        base::Vector2,
        geometry::{
            Point2,
            Point3,
            Rotation2,
            Scale2,
        },
    },
    ::std::{
        io::stderr,
        ops::Range,
        process::ExitCode,
        slice::from_raw_parts as new_slice,
        sync::{
            Arc,
            mpsc::channel,
        },
        time::Instant,
    },
    ::tracing::{
        debug,
        error,
        info,
        instrument,
        level_filters::LevelFilter,
        trace,
        warn,
    },
    ::tracing_subscriber::{
        filter::EnvFilter,
        fmt::Layer as FmtLayer,
        layer::SubscriberExt as _,
        registry::Registry,
        util::SubscriberInitExt as _,
    },
    ::wgpu::{
        BackendOptions,
        Backends,
        BlendState,
        Buffer,
        BufferAddress,
        BufferDescriptor,
        BufferUsages,
        Color,
        ColorTargetState,
        ColorWrites,
        CommandEncoderDescriptor,
        Device,
        DeviceDescriptor,
        Features,
        FragmentState,
        FrontFace,
        Instance,
        InstanceDescriptor,
        InstanceFlags,
        Limits,
        LoadOp,
        MapMode,
        MemoryHints,
        MultisampleState,
        NoopBackendOptions,
        Operations,
        PUSH_CONSTANT_ALIGNMENT,
        PipelineLayoutDescriptor,
        PollType,
        PolygonMode,
        PowerPreference,
        PrimitiveState,
        PrimitiveTopology,
        PushConstantRange,
        Queue,
        RenderPassColorAttachment,
        RenderPassDescriptor,
        RenderPipeline,
        RenderPipelineDescriptor,
        RequestAdapterOptions,
        ShaderStages,
        StoreOp,
        Surface,
        SurfaceConfiguration,
        TextureAspect,
        TextureViewDescriptor,
        TextureViewDimension,
        Trace,
        VERTEX_STRIDE_ALIGNMENT,
        VertexAttribute,
        VertexBufferLayout,
        VertexState,
        VertexStepMode,
        include_wgsl,
        vertex_attr_array,
    },
    ::winit::{
        application::ApplicationHandler,
        dpi::{
            PhysicalPosition,
            PhysicalSize,
        },
        event::{
            ElementState,
            KeyEvent,
            MouseButton,
            WindowEvent,
        },
        event_loop::{
            ActiveEventLoop,
            ControlFlow,
            EventLoop,
        },
        keyboard::{
            Key,
            NamedKey,
        },
        window::{
            Window,
            WindowId,
        },
    },
};

fn main() -> ExitCode {
    match App::report(App::run()) {
        true => ExitCode::SUCCESS,
        false => ExitCode::FAILURE,
    }
}

struct App {
    args: Args,
    context: Option<Context>,
}

impl App {
    fn run() -> Result<()> {
        install_eyre()?;
        let args = Args::parse();

        Registry::default()
            .with(FmtLayer::new().with_writer(stderr))
            .with(
                EnvFilter::builder()
                    .with_default_directive(LevelFilter::INFO.into())
                    .parse(&args.tracing_directives)?,
            )
            .try_init()?;

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = Self {
            args,
            context: Option::None,
        };

        event_loop.run_app(&mut app)?;
        Result::Ok(())
    }

    fn report(result: Result<()>) -> bool {
        match result {
            Result::Ok(()) => true,
            Result::Err(error) => {
                error!(error = &*error);
                false
            },
        }
    }

    fn handle_result(result: Result<()>) {
        if !Self::report(result) {
            panic!();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.context.is_some() {
            return;
        }
        Self::handle_result(
            Context::new(
                event_loop,
                self.args.max_bones,
                self.args.ik_steps,
                self.args.ik_sens,
                PALETTE[self.args.theme_flavor].colors,
            )
            .map(|context| self.context = Option::Some(context)),
        );
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        Self::handle_result(
            self.context
                .as_mut()
                .ok_or_eyre("context is none")
                .and_then(|context| context.handle_window_event(event_loop, window_id, event)),
        );
    }
}

#[derive(Parser)]
struct Args {
    #[clap(short = 'b', long, default_value_t = 8)]
    max_bones: usize,
    #[clap(short = 'n', long, default_value_t = 10)]
    ik_steps: u32,
    #[clap(short = 's', long, default_value_t = 0.2)]
    ik_sens: f32,
    #[clap(short = 'f', long, default_value = "latte")]
    theme_flavor: FlavorName,
    #[clap(short = 't', long, default_value = "")]
    tracing_directives: String,
}

struct Context {
    window: Arc<Window>,
    cursor: PhysicalPosition<f64>,
    mode: Mode,
    root: Option<Point2<f32>>,
    bones: Bones,
    max_bones: usize,
    ik_steps: u32,
    ik_sens: f32,
    renderer: Renderer,
    rects: Vec<Rect>,
    bone_color: Point3<f32>,
    linkage_color: Point3<f32>,
    end_linkage_color: Point3<f32>,
    cursor_color: Point3<f32>,
    start_instant: Instant,
    render_count: u32,
}

impl Context {
    #[instrument(skip(flavor_colors))]
    fn new(
        event_loop: &ActiveEventLoop,
        max_bones: usize,
        ik_steps: u32,
        ik_sens: f32,
        flavor_colors: FlavorColors,
    ) -> Result<Self> {
        let window = Arc::new(event_loop.create_window(Default::default())?);
        trace!("{window:?}");
        let cursor = Default::default();
        let mode = Mode::Edit;
        let root = Option::None;
        let bones = Bones::new(max_bones);
        let max_rects = 2 * (max_bones + 1);
        let renderer = Renderer::new(window.clone(), max_rects, Self::color(flavor_colors.base))?;
        let rects = vec![Default::default(); max_rects];
        let bone_color = Self::color(flavor_colors.text);
        let linkage_color = Self::color(flavor_colors.lavender);
        let end_linkage_color = Self::color(flavor_colors.sky);
        let cursor_color = Self::color(flavor_colors.maroon);
        let start_instant = Instant::now();
        let render_count = 0;

        Result::Ok(Self {
            window,
            cursor,
            mode,
            root,
            bones,
            max_bones,
            ik_steps,
            ik_sens,
            renderer,
            rects,
            bone_color,
            linkage_color,
            end_linkage_color,
            cursor_color,
            start_instant,
            render_count,
        })
    }

    #[instrument(skip(self))]
    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<()> {
        if self.window.id() != window_id {
            warn!("unknown window");
            return Result::Ok(());
        }

        match event {
            WindowEvent::Resized(PhysicalSize {
                width,
                height,
            }) => self.renderer.configure(Point2::new(width, height)),
            WindowEvent::CloseRequested => {
                info!("exit event loop");
                event_loop.exit();
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(named_key),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match named_key {
                NamedKey::Space => {
                    self.mode = match self.mode {
                        Mode::Edit => Mode::Follow,
                        Mode::Follow => Mode::Edit,
                    }
                },
                NamedKey::Backspace if matches!(self.mode, Mode::Edit) => {
                    if self.bones.pop().is_none() && self.root.take().is_none() {
                        debug!("there is no linkage");
                    }
                },
                _ => {},
            },
            WindowEvent::CursorMoved {
                position, ..
            } => self.cursor = position,
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if matches!(self.mode, Mode::Edit) => match self.bones.len() < self.max_bones {
                true => match self.root {
                    Option::Some(root) => self.bones.push(self.bones.as_bone(root, self.cursor())),
                    Option::None => self.root = Option::Some(self.cursor()),
                },
                false => debug!("buffer is full"),
            },
            WindowEvent::RedrawRequested => {
                if let Option::Some(root) = self.root {
                    let cursor = self.cursor();

                    if matches!(self.mode, Mode::Follow) {
                        for _ in 0..self.ik_steps {
                            self.bones
                                .ik(self.bones.as_bone(root, cursor), self.ik_sens);
                        }
                    }

                    let node_rect = Rect {
                        size: Point2::from(Vector2::repeat(0.02)),
                        angle: 0.0,
                        center: Point2::origin(),
                        color: self.linkage_color,
                    };

                    self.rects[self.bones.len()] = Rect {
                        center: root,
                        ..node_rect
                    };

                    let mut p0 = root;

                    for (i, (p1, angle)) in self.bones.fk(root).enumerate() {
                        self.rects[i] = Rect {
                            size: Point2::new((p1 - p0).norm(), 0.005),
                            angle,
                            center: Point2::from((p0.coords + p1.coords) / 2.0),
                            color: self.bone_color,
                        };

                        self.rects[self.bones.len() + 1 + i] = Rect {
                            center: p1,
                            ..node_rect
                        };

                        p0 = p1;
                    }

                    self.rects[2 * self.bones.len()].color = self.end_linkage_color;

                    self.rects[2 * self.bones.len() + 1] = Rect {
                        center: cursor,
                        color: self.cursor_color,
                        ..node_rect
                    };
                }

                self.renderer.render(
                    &self.rects,
                    0,
                    0..match self.root {
                        Option::Some(_) => 2 * (self.bones.len() + 1),
                        Option::None => 0,
                    },
                )?;

                self.window.request_redraw();
                self.render_count += 1;

                let elapsed = self.start_instant.elapsed().as_secs_f32();

                if elapsed >= 1.0 {
                    info!(fps = self.render_count as f32 / elapsed);
                    self.start_instant = Instant::now();
                    self.render_count = 0;
                }
            },
            _ => (),
        }

        Result::Ok(())
    }

    fn cursor(&self) -> Point2<f32> {
        let window_size = self.window.inner_size();
        let window_size = Vector2::new(window_size.width, window_size.height).cast::<f32>();

        Point2::from(
            (2.0 * Vector2::new(self.cursor.x, self.cursor.y).cast::<f32>() - window_size)
                .component_mul(&Vector2::new(1.0, -1.0))
                / window_size.min(),
        )
    }

    fn color(ccolor: CColor) -> Point3<f32> {
        Point3::new(ccolor.rgb.r, ccolor.rgb.g, ccolor.rgb.b).cast() / u8::MAX as _
    }
}

enum Mode {
    Edit,
    Follow,
}

struct Bones {
    inner: Vec<Vector2<f32>>,
}

impl Bones {
    fn new(capacity: usize) -> Self {
        Self {
            inner: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, bone: Vector2<f32>) {
        self.inner.push(bone);
    }

    fn pop(&mut self) -> Option<Vector2<f32>> {
        self.inner.pop()
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn fk(&self, mut root: Point2<f32>) -> impl Iterator<Item = (Point2<f32>, f32)> {
        let mut angle = 0.0;

        self.inner.iter().copied().map(move |bone| {
            let vector = Rotation2::new(angle) * bone;
            root += vector;
            angle = Self::angle(vector);
            (root, angle)
        })
    }

    fn ik(&mut self, mut target: Vector2<f32>, sens: f32) {
        let mut end_effector = Vector2::zeros();

        for bone in self.inner.iter_mut().rev() {
            end_effector = *bone + Rotation2::new(Self::angle(*bone)) * end_effector;
            target = *bone + Rotation2::new(Self::angle(*bone)) * target;
            let angle = sens * Self::angle_to(end_effector, target);
            *bone = Rotation2::new(angle) * *bone;
        }
    }

    fn as_bone(&self, root: Point2<f32>, end: Point2<f32>) -> Vector2<f32> {
        let (vector, angle) = self.fk(root).last().unwrap_or((root, 0.0));
        Rotation2::new(angle).inverse() * (end - vector)
    }

    fn angle_to(v0: Vector2<f32>, v1: Vector2<f32>) -> f32 {
        f32::atan2(v0.perp(&v1), v0.dot(&v1))
    }

    fn angle(v: Vector2<f32>) -> f32 {
        Self::angle_to(Vector2::new(1.0, 0.0), v)
    }
}

struct Renderer {
    surface: Surface<'static>,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,
    constant_buffer: Vec<u8>,
    constant_layout: StructLayout,
    staging_buffer: Buffer,
    vertex_buffer: Buffer,
    vertex_layout: StructLayout,
    pipeline: RenderPipeline,
    clear_color: Color,
}

impl Renderer {
    #[instrument]
    fn new(window: Arc<Window>, max_rects: usize, clear_color: Point3<f32>) -> Result<Self> {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::PRIMARY,
            flags: match cfg!(debug_assertions) {
                true => InstanceFlags::debugging(),
                false => InstanceFlags::empty(),
            },
            backend_options: BackendOptions {
                gl: Default::default(),
                dx12: Default::default(),
                noop: NoopBackendOptions {
                    enable: false,
                },
            },
        });

        trace!("{instance:?}");
        let surface = instance.create_surface(window)?;
        trace!("{surface:?}");

        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Option::Some(&surface),
        }))?;

        trace!("{adapter:?}");

        let surface_config = surface
            .get_default_config(&adapter, 0, 0)
            .ok_or_eyre("adapter does not support surface")?;

        let constant_layout = StructLayout::new_constant(&[size_of::<Vector2<f32>>()]);

        let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
            label: Option::None,
            required_features: Features::PUSH_CONSTANTS,
            required_limits: Limits {
                max_push_constant_size: constant_layout.size as _,
                ..Default::default()
            },
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        }))?;

        trace!("{device:?}");
        trace!("{queue:?}");

        let vertex_attributes = vertex_attr_array![
            0 => Float32x2,
            1 => Float32x2,
            2 => Float32x2,
            3 => Float32x3,
        ];

        let vertex_layout = StructLayout::new_vertex(&vertex_attributes);
        let buffer_size = max_rects as BufferAddress * vertex_layout.size as BufferAddress;

        let staging_buffer = device.create_buffer(&BufferDescriptor {
            label: Option::None,
            size: buffer_size,
            usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        trace!("{staging_buffer:?}");

        let vertex_buffer = device.create_buffer(&BufferDescriptor {
            label: Option::None,
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        trace!("{vertex_buffer:?}");

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Option::None,
            bind_group_layouts: &[],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::VERTEX,
                range: 0..constant_layout.size as _,
            }],
        });

        trace!("{pipeline_layout:?}");
        let module = device.create_shader_module(include_wgsl!("shader.wgsl"));
        trace!("{module:?}");

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Option::None,
            layout: Option::Some(&pipeline_layout),
            vertex: VertexState {
                module: &module,
                entry_point: Option::None,
                compilation_options: Default::default(),
                buffers: &[VertexBufferLayout {
                    array_stride: vertex_layout.size as _,
                    step_mode: VertexStepMode::Instance,
                    attributes: &vertex_attributes,
                }],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                strip_index_format: Option::None,
                front_face: FrontFace::Ccw,
                cull_mode: Option::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Option::None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Option::Some(FragmentState {
                module: &module,
                entry_point: Option::None,
                compilation_options: Default::default(),
                targets: &[Option::Some(ColorTargetState {
                    format: surface_config.format,
                    blend: Option::Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::all(),
                })],
            }),
            multiview: Option::None,
            cache: Option::None,
        });

        trace!("{pipeline:?}");
        let constant_buffer = vec![0; constant_layout.size];
        let clear_color = clear_color.cast();

        let clear_color = Color {
            r: clear_color.x,
            g: clear_color.y,
            b: clear_color.z,
            a: 1.0,
        };

        Result::Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            constant_buffer,
            constant_layout,
            staging_buffer,
            vertex_buffer,
            vertex_layout,
            pipeline,
            clear_color,
        })
    }

    #[instrument(skip(self))]
    fn configure(&mut self, size: Point2<u32>) {
        self.surface_config.width = size.x;
        self.surface_config.height = size.y;

        if size.iter().copied().all(|x| x > 0) {
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    #[instrument(skip(self, update_rects))]
    fn render(
        &mut self,
        update_rects: &[Rect],
        update_offset: usize,
        render_rects: Range<usize>,
    ) -> Result<()> {
        if self.surface_config.width == 0 || self.surface_config.height == 0 {
            return Result::Ok(());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Option::None,
            });

        trace!("{encoder:?}");

        if !update_rects.is_empty() {
            let staging_slice = self
                .staging_buffer
                .slice(..(update_rects.len() * self.vertex_layout.size) as BufferAddress);

            trace!("{staging_slice:?}");
            let (sender, receiver) = channel();

            staging_slice.map_async(MapMode::Write, move |result| {
                debug!("receive a result of map_async");

                if let Result::Err(error) = sender
                    .send(result)
                    .wrap_err("cannot send a result from a callback")
                {
                    error!(error = &*error);
                }
            });

            self.device.poll(PollType::Wait)?;
            receiver.recv()??;
            let mut staging_view = staging_slice.get_mapped_range_mut();

            for (index, rect) in update_rects.iter().enumerate() {
                for (field, point) in [Point2::new(1.0, 0.0), Point2::new(0.0, 1.0)]
                    .into_iter()
                    .enumerate()
                {
                    self.vertex_layout.write(
                        &mut staging_view,
                        index,
                        field,
                        &(Rotation2::new(rect.angle)
                            * (Scale2::from(0.5 * rect.size.coords) * point))
                            .coords,
                    );
                }

                self.vertex_layout
                    .write(&mut staging_view, index, 2, &rect.center.coords);

                self.vertex_layout
                    .write(&mut staging_view, index, 3, &rect.color.coords);
            }

            drop(staging_view);
            self.staging_buffer.unmap();

            encoder.copy_buffer_to_buffer(
                &self.staging_buffer,
                0,
                &self.vertex_buffer,
                (update_offset * self.vertex_layout.size) as _,
                (update_rects.len() * self.vertex_layout.size) as _,
            );
        }

        let surface_texture = self.surface.get_current_texture()?;
        trace!("{surface_texture:?}");

        let surface_view = surface_texture.texture.create_view(&TextureViewDescriptor {
            label: Option::None,
            format: Option::Some(self.surface_config.format),
            dimension: Option::Some(TextureViewDimension::D2),
            usage: Option::None,
            aspect: TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Option::None,
            base_array_layer: 0,
            array_layer_count: Option::None,
        });

        trace!("{surface_view:?}");

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Option::None,
            color_attachments: &[Option::Some(RenderPassColorAttachment {
                view: &surface_view,
                resolve_target: Option::None,
                ops: Operations {
                    load: LoadOp::Clear(self.clear_color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Option::None,
            timestamp_writes: Option::None,
            occlusion_query_set: Option::None,
        });

        trace!("{pass:?}");

        if !render_rects.is_empty() {
            pass.set_pipeline(&self.pipeline);

            let window_size =
                Vector2::new(self.surface_config.width, self.surface_config.height).cast::<f32>();

            self.constant_layout.write(
                &mut self.constant_buffer,
                0,
                0,
                &Vector2::repeat(window_size.min()).component_div(&window_size),
            );

            pass.set_push_constants(
                ShaderStages::VERTEX,
                0,
                &self.constant_buffer[..self.constant_layout.size],
            );

            pass.set_vertex_buffer(
                0,
                self.vertex_buffer.slice(
                    (render_rects.start * self.vertex_layout.size) as BufferAddress
                        ..(render_rects.end * self.vertex_layout.size) as _,
                ),
            );

            pass.draw(0..4, 0..(render_rects.end - render_rects.start) as _);
        }

        drop(pass);
        self.queue.submit([encoder.finish()]);
        surface_texture.present();
        Result::Ok(())
    }
}

struct StructLayout {
    size: usize,
    fields: Vec<StructFieldLayout>,
}

impl StructLayout {
    fn new<T: Copy>(
        fields: &[T],
        f: impl Fn(usize, T) -> (usize, StructFieldLayout),
        alignment: usize,
    ) -> Self {
        let mut struct_size = 0;
        let mut struct_fields = Vec::with_capacity(fields.len());

        for field in fields.iter().copied() {
            let field_layout;
            (struct_size, field_layout) = f(struct_size, field);
            struct_fields.push(field_layout);
        }

        Self {
            size: struct_size.div_ceil(alignment) * alignment,
            fields: struct_fields,
        }
    }

    fn new_constant(sizes: &[usize]) -> Self {
        Self::new(
            sizes,
            |struct_size, field_size| {
                (
                    struct_size + field_size,
                    StructFieldLayout {
                        offset: struct_size,
                        size: field_size,
                    },
                )
            },
            PUSH_CONSTANT_ALIGNMENT as _,
        )
    }

    fn new_vertex(attributes: &[VertexAttribute]) -> Self {
        Self::new(
            attributes,
            |struct_size, attribute| {
                let field_offset = attribute.offset as _;
                let field_size = attribute.format.size() as _;

                (
                    struct_size.max(field_offset + field_size),
                    StructFieldLayout {
                        offset: field_offset,
                        size: field_size,
                    },
                )
            },
            VERTEX_STRIDE_ALIGNMENT as _,
        )
    }

    fn write<T>(&self, buffer: &mut [u8], index: usize, field: usize, value: &T) {
        buffer[index * self.size..][self.fields[field].offset..][..self.fields[field].size]
            .copy_from_slice(unsafe { new_slice(value as *const _ as *const _, size_of::<T>()) });
    }
}

struct StructFieldLayout {
    offset: usize,
    size: usize,
}

#[derive(Clone, Copy, Default)]
struct Rect {
    size: Point2<f32>,
    angle: f32,
    center: Point2<f32>,
    color: Point3<f32>,
}
