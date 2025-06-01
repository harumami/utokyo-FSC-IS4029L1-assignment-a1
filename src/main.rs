use {
    ::catppuccin::{
        Color as CColor,
        PALETTE,
    },
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
    },
    ::tracing::{
        error,
        info,
        instrument,
        metadata::Level,
        trace,
        warn,
    },
    ::tracing_subscriber::{
        fmt::Subscriber,
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
    catppuccin::FlavorColors,
};

fn main() -> ExitCode {
    match App::report(App::run()) {
        true => ExitCode::SUCCESS,
        false => ExitCode::FAILURE,
    }
}

struct App {
    context: Option<Context>,
}

impl App {
    fn run() -> Result<()> {
        install_eyre()?;

        Subscriber::builder()
            .with_max_level(Level::TRACE)
            .with_writer(stderr)
            .finish()
            .try_init()?;

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = Self {
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
        if self.context.is_none() {
            Self::handle_result(
                Context::new(event_loop, 16, PALETTE.latte.colors)
                    .map(|context| self.context = Option::Some(context)),
            );
        }
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

struct Context {
    linkages: Vec<Vector2<f32>>,
    max_linkages: usize,
    window: Arc<Window>,
    cursor: PhysicalPosition<f64>,
    renderer: Renderer,
    rects: Vec<Rect>,
    arm_color: Point3<f32>,
}

impl Context {
    #[instrument]
    fn new(
        event_loop: &ActiveEventLoop,
        max_linkages: usize,
        flavor_colors: FlavorColors,
    ) -> Result<Self> {
        let window = Arc::new(event_loop.create_window(Default::default())?);
        trace!("{window:?}");
        let max_rects = 2 * max_linkages - 1;
        let renderer = Renderer::new(window.clone(), max_rects, Self::color(flavor_colors.base))?;
        let linkages = Vec::with_capacity(max_linkages);
        let cursor = Default::default();
        let rects = Vec::with_capacity(max_rects);
        let arm_color = Self::color(flavor_colors.text);

        Result::Ok(Self {
            linkages,
            max_linkages,
            window,
            cursor,
            renderer,
            rects,
            arm_color,
        })
    }

    #[instrument(skip(self))]
    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<()> {
        match self.window.id() == window_id {
            true => match event {
                WindowEvent::Resized(PhysicalSize {
                    width,
                    height,
                }) => self.renderer.configure(Scale2::new(width, height)),
                WindowEvent::CloseRequested => {
                    info!("exit event loop");
                    event_loop.exit();
                },
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            logical_key: Key::Named(NamedKey::Backspace),
                            state: ElementState::Pressed,
                            repeat: false,
                            ..
                        },
                    ..
                } => match self.linkages.is_empty() {
                    true => info!("there are no linkages"),
                    false => {
                        self.linkages.pop();
                    },
                },
                WindowEvent::CursorMoved {
                    position, ..
                } => self.cursor = position,
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => match self.linkages.len() < self.max_linkages {
                    true => {
                        let window_size = self.window.inner_size();

                        let window_size =
                            Vector2::new(window_size.width, window_size.height).cast::<f32>();

                        let position = (2.0
                            * Vector2::new(self.cursor.x, self.cursor.y).cast::<f32>()
                            - window_size)
                            .component_mul(&Vector2::new(1.0, -1.0))
                            / window_size.min();

                        self.linkages.push(position);
                    },
                    false => info!("buffer is full"),
                },
                WindowEvent::RedrawRequested => {
                    for linkage in &self.linkages {
                        self.rects.push(Rect {
                            size: Scale2::new(0.03, 0.03),
                            angle: Rotation2::new(0.0),
                            center: Point2::from(*linkage),
                            color: self.arm_color,
                        })
                    }

                    self.renderer.render(&self.rects, 0, 0..self.rects.len())?;
                    self.rects.clear();
                    self.window.request_redraw();
                },
                _ => (),
            },
            false => warn!("unknown window"),
        }

        Result::Ok(())
    }

    fn color(ccolor: CColor) -> Point3<f32> {
        let rgb = ccolor.rgb;
        Point3::new(rgb.r, rgb.g, rgb.b).cast() / u8::MAX as _
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
    fn configure(&mut self, size: Scale2<u32>) {
        let size = size * Point2::new(1, 1);

        if size.iter().copied().all(|x| x > 0) {
            self.surface_config.width = size.x;
            self.surface_config.height = size.y;
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
                info!("receive a result of map_async");

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
                let scale = rect.size * 0.5;

                for (field, point) in [Point2::new(1.0, 0.0), Point2::new(0.0, 1.0)]
                    .into_iter()
                    .enumerate()
                {
                    self.vertex_layout.write(
                        &mut staging_view,
                        index,
                        field,
                        &(rect.angle * (scale * point)),
                    );
                }

                self.vertex_layout
                    .write(&mut staging_view, index, 2, &rect.center);

                self.vertex_layout
                    .write(&mut staging_view, index, 3, &rect.color);
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

struct Rect {
    size: Scale2<f32>,
    angle: Rotation2<f32>,
    center: Point2<f32>,
    color: Point3<f32>,
}
