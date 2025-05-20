use {
    color_eyre::eyre::Result,
    std::{
        process::ExitCode,
        sync::Arc,
    },
    tracing::error,
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
    App::handle_result(App::run())
}

struct App {
    context: Option<Context>,
}

struct Context {
    window: Arc<Window>,
}

impl App {
    fn run() -> Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut app = Self {
            context: Option::None,
        };

        event_loop.run_app(&mut app)?;
        Result::Ok(())
    }

    fn resume(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        if self.context.is_none() {
            let window = Arc::new(event_loop.create_window(Default::default())?);

            self.context = Option::Some(Context {
                window,
            });
        }

        Result::Ok(())
    }

    fn handle_window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) -> Result<()> {
        let Option::Some(context) = self.context.as_ref() else {
            return Result::Ok(());
        };

        if context.window.id() != window_id {
            return Result::Ok(());
        }

        _ = event_loop;
        _ = event;
        Result::Ok(())
    }

    fn handle_result(result: Result<()>) -> ExitCode {
        match result {
            Result::Ok(()) => ExitCode::SUCCESS,
            Result::Err(error) => {
                error!(error = &*error);
                ExitCode::FAILURE
            },
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
