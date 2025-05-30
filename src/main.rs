use {
    color_eyre::{
        eyre::{
            OptionExt as _,
            Result,
        },
        install as install_eyre,
    },
    futures::executor::block_on,
    std::{
        io::stderr,
        process::ExitCode,
        sync::Arc,
    },
    tracing::{
        error,
        info,
        instrument,
        metadata::Level,
        trace,
        warn,
    },
    tracing_subscriber::{
        fmt::{
            Subscriber,
            format::FmtSpan,
        },
        util::SubscriberInitExt as _,
    },
    wgpu::{
        BackendOptions,
        Backends,
        BlendState,
        Color,
        ColorTargetState,
        ColorWrites,
        CommandEncoderDescriptor,
        Device,
        DeviceDescriptor,
        Face,
        Features,
        FragmentState,
        FrontFace,
        Instance,
        InstanceDescriptor,
        InstanceFlags,
        LoadOp,
        MemoryHints,
        MultisampleState,
        NoopBackendOptions,
        Operations,
        PolygonMode,
        PowerPreference,
        PrimitiveState,
        PrimitiveTopology,
        Queue,
        RenderPassColorAttachment,
        RenderPassDescriptor,
        RenderPipeline,
        RenderPipelineDescriptor,
        RequestAdapterOptions,
        StoreOp,
        Surface,
        SurfaceConfiguration,
        TextureAspect,
        TextureViewDescriptor,
        TextureViewDimension,
        Trace,
        VERTEX_STRIDE_ALIGNMENT,
        VertexBufferLayout,
        VertexState,
        VertexStepMode,
        include_wgsl,
        vertex_attr_array,
    },
    winit::{
        application::ApplicationHandler,
        event::WindowEvent,
        event_loop::{
            ActiveEventLoop,
            ControlFlow,
            EventLoop,
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
    context: Option<Context>,
}

struct Context {
    window: Arc<Window>,
    surface: Surface<'static>,
    device: Device,
    queue: Queue,
    surface_config: SurfaceConfiguration,
    pipeline: RenderPipeline,
}

impl App {
    fn run() -> Result<()> {
        install_eyre()?;

        Subscriber::builder()
            .with_max_level(Level::TRACE)
            .with_writer(stderr)
            .with_span_events(FmtSpan::ACTIVE)
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

    #[instrument(skip(self))]
    fn resume(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        if self.context.is_none() {
            let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
            trace!("{window:?}");

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
            let surface = instance.create_surface(window.clone())?;
            trace!("{surface:?}");

            let adapter = block_on(instance.request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Option::Some(&surface),
            }))?;

            trace!("{adapter:?}");

            let (device, queue) = block_on(adapter.request_device(&DeviceDescriptor {
                label: Option::None,
                required_features: Features::empty(),
                required_limits: Default::default(),
                memory_hints: MemoryHints::Performance,
                trace: Trace::Off,
            }))?;

            trace!("{device:?}");
            trace!("{queue:?}");

            let surface_config = surface
                .get_default_config(&adapter, 0, 0)
                .ok_or_eyre("adapter does not support surface")?;

            let module = device.create_shader_module(include_wgsl!("shader.wgsl"));
            trace!("{module:?}");

            let vertex_attributes = vertex_attr_array![
                0 => Float32x2,
            ];

            let vertex_size = vertex_attributes
                .iter()
                .map(|attribute| attribute.offset + attribute.format.size())
                .max()
                .unwrap_or(0)
                .div_ceil(VERTEX_STRIDE_ALIGNMENT)
                * VERTEX_STRIDE_ALIGNMENT;

            let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Option::None,
                layout: Option::None,
                vertex: VertexState {
                    module: &module,
                    entry_point: Option::None,
                    compilation_options: Default::default(),
                    buffers: &[VertexBufferLayout {
                        array_stride: vertex_size,
                        step_mode: VertexStepMode::Vertex,
                        attributes: &vertex_attributes,
                    }],
                },
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::LineList,
                    strip_index_format: Option::None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Option::Some(Face::Back),
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

            let context = Context {
                window,
                surface,
                device,
                queue,
                surface_config,
                pipeline,
            };

            self.context = Option::Some(context);
        }

        Result::Ok(())
    }

    #[instrument(skip(self))]
    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<()> {
        match self.context.as_ref() {
            Option::Some(Context {
                window,
                surface,
                device,
                queue,
                surface_config,
                pipeline,
            }) => match window.id() == window_id {
                true => match event {
                    WindowEvent::Resized(size) if size.width != 0 && size.height != 0 => {
                        surface.configure(
                            device,
                            &SurfaceConfiguration {
                                width: size.width,
                                height: size.height,
                                ..surface_config.clone()
                            },
                        );
                    },
                    WindowEvent::CloseRequested => {
                        info!("exit event loop");
                        event_loop.exit();
                    },
                    WindowEvent::RedrawRequested => {
                        let surface_texture = surface.get_current_texture()?;

                        let mut encoder =
                            device.create_command_encoder(&CommandEncoderDescriptor {
                                label: Option::None,
                            });

                        trace!("{encoder:?}");

                        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                            label: Option::None,
                            color_attachments: &[Option::Some(RenderPassColorAttachment {
                                view: &surface_texture.texture.create_view(
                                    &TextureViewDescriptor {
                                        label: Option::None,
                                        format: Option::Some(surface_config.format),
                                        dimension: Option::Some(TextureViewDimension::D2),
                                        usage: Option::None,
                                        aspect: TextureAspect::All,
                                        base_mip_level: 0,
                                        mip_level_count: Option::None,
                                        base_array_layer: 0,
                                        array_layer_count: Option::None,
                                    },
                                ),
                                resolve_target: Option::None,
                                ops: Operations {
                                    load: LoadOp::Clear(Color::BLACK),
                                    store: StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Option::None,
                            timestamp_writes: Option::None,
                            occlusion_query_set: Option::None,
                        });

                        trace!("{pass:?}");
                        pass.set_pipeline(pipeline);
                        // pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                        // pass.draw(0..vertex_count as u32, 0..1);
                        drop(pass);
                        queue.submit([encoder.finish()]);
                        surface_texture.present();
                    },
                    _ => (),
                },
                false => warn!("unknown window"),
            },
            Option::None => warn!("context is none"),
        }

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
        Self::handle_result(self.resume(event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        Self::handle_result(self.handle_window_event(event_loop, window_id, event));
    }
}
