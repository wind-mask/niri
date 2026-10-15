#[macro_use]
extern crate tracing;

use std::fmt::Write as _;
use std::fs::{create_dir_all, File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::FromRawFd;
use std::os::unix::fs::MetadataExt as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::sync::Mutex;
use std::{env, mem};

use calloop::EventLoop;
use clap::{CommandFactory, Parser};
use clap_complete::Shell;
use clap_complete_nushell::Nushell;
use directories::ProjectDirs;
use niri::cli::{Cli, CompletionShell, Sub};
#[cfg(feature = "dbus")]
use niri::dbus;
use niri::ipc::client::handle_msg;
use niri::niri::State;
use niri::utils::spawning::{
    spawn, spawn_sh, store_and_increase_nofile_rlimit, CHILD_DISPLAY, CHILD_ENV,
    REMOVE_ENV_RUST_BACKTRACE, REMOVE_ENV_RUST_LIB_BACKTRACE,
};
use niri::utils::{cause_panic, version, watcher, xwayland, IS_SYSTEMD_SERVICE};
use niri_config::{Config, ConfigPath};
use niri_ipc::socket::SOCKET_PATH_ENV;
use sd_notify::NotifyState;
use smithay::reexports::wayland_server::Display;
use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;
use tracing_subscriber::{EnvFilter, Layer as _};

const DEFAULT_LOG_FILTER: &str =
    "niri=debug,niri::render_helpers::gpu=off,smithay::backend::renderer::gles=error";

/// Environment variable naming the file the GPU and upscaling diagnostics are written to.
///
/// These diagnostics are far too verbose for the journal, which drops messages when they arrive
/// too quickly, so they get a file of their own. Setting this when starting niri is how the
/// Hybrid work is measured; the rest of the logging still goes to the standard error.
const GPU_LOG_PATH_ENV: &str = "NIRI_GPU_LOG";

/// Environment variable overriding the filter of the dedicated diagnostics file.
const GPU_LOG_FILTER_ENV: &str = "NIRI_GPU_LOG_FILTER";

/// The filter of the dedicated file.
///
/// What is here is the state the diagnostics report once: the route a frame took, which pass ran,
/// why a window fell back. The output that repeats for every pass, every window and every frame is
/// TRACE, so a report that needs it turns the targets up with this variable, at the cost of a lot
/// of output: with `NIRI_GPU_LOG_FILTER`, set `niri::render_helpers::gpu=trace` and
/// `niri::render_helpers::superres=trace` to get everything.
const DEFAULT_GPU_LOG_FILTER: &str =
    "niri::render_helpers::gpu=debug,niri::render_helpers::superres=debug";

/// Environment variable setting how large the diagnostics file may grow.
///
/// The file is written for every window and every pass, so it grows without a bound of its own;
/// the most recent part of it is what answers "what is happening now", so the limit keeps that and
/// drops what came before. Takes a byte count, optionally with a `K`, `M` or `G` suffix, and `off`
/// to keep everything.
const GPU_LOG_MAX_BYTES_ENV: &str = "NIRI_GPU_LOG_MAX_BYTES";

/// How much of the diagnostics file is kept when [`GPU_LOG_MAX_BYTES_ENV`] is not set.
const DEFAULT_GPU_LOG_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// Where one of the logging layers writes.
type LogWriter = Mutex<Box<dyn Write + Send>>;

/// Sets up logging: the standard error, and the file the GPU diagnostics go to when configured.
fn init_logging() {
    let directives = env::var("RUST_LOG").unwrap_or_else(|_| DEFAULT_LOG_FILTER.to_owned());
    let path = env::var_os(GPU_LOG_PATH_ENV).map(PathBuf::from);
    let (limit, limit_error) = gpu_log_limit();

    let (gpu_writer, gpu_error) = match gpu_log_file(path.as_deref(), limit) {
        Ok(writer) => (writer, None),
        Err(err) => (Box::new(io::sink()) as Box<dyn Write + Send>, Some(err)),
    };
    let gpu_log = path.is_some() && gpu_error.is_none();

    // The file takes over the diagnostics it is configured for, so that the standard error does
    // not get them twice. Without a file the layer stays, writing nowhere.
    let gpu_filter = if gpu_log {
        env::var(GPU_LOG_FILTER_ENV).unwrap_or_else(|_| DEFAULT_GPU_LOG_FILTER.to_owned())
    } else {
        "off".to_owned()
    };

    subscriber(
        Mutex::new(Box::new(io::stderr()) as Box<dyn Write + Send>),
        &stderr_directives(&directives, gpu_log),
        Mutex::new(gpu_writer),
        &gpu_filter,
    )
    .init();

    // The expensive parts of the diagnostics only run when something records them.
    niri::render_helpers::renderer::set_gpu_diagnostics(
        gpu_log || diagnostics_asked_for(&directives),
    );

    if let Some(error) = limit_error {
        warn!(
            "ignoring {GPU_LOG_MAX_BYTES_ENV}: {error}; the file keeps the most recent {}",
            format_size(DEFAULT_GPU_LOG_MAX_BYTES)
        );
    }

    if let Some(err) = gpu_error {
        warn!("cannot open the GPU log file: {err}; the GPU diagnostics are not written anywhere");
    } else if gpu_log {
        if let Some(path) = path {
            info!(
                "writing the GPU and upscaling diagnostics to {} ({})",
                path.display(),
                match limit {
                    0 => "keeping everything".to_owned(),
                    limit => format!("keeping the most recent {}", format_size(limit)),
                }
            );
        }
    }
}

/// Builds the logging subscriber: one layer for the standard error, one for the diagnostics.
///
/// Each layer has its own filter, so that the diagnostics have a destination of their own instead
/// of being mixed into the session log.
fn subscriber(
    stderr_writer: LogWriter,
    stderr_directives: &str,
    gpu_writer: LogWriter,
    gpu_directives: &str,
) -> impl tracing::Subscriber + Send + Sync + 'static {
    let stderr_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_ansi_sanitization(false)
        .with_writer(stderr_writer)
        .with_filter(EnvFilter::builder().parse_lossy(stderr_directives.to_owned()));
    let gpu_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(gpu_writer)
        .with_filter(EnvFilter::builder().parse_lossy(gpu_directives.to_owned()));

    tracing_subscriber::registry()
        .with(stderr_layer)
        .with(gpu_layer)
}

/// Returns whether the filter names a diagnostic target at a level that records it.
///
/// This is what decides whether the expensive parts of the diagnostics run when there is no file.
fn diagnostics_asked_for(directives: &str) -> bool {
    const TARGETS: [&str; 2] = [
        "niri::render_helpers::gpu",
        "niri::render_helpers::superres",
    ];

    directives.split(',').any(|directive| {
        let Some((target, level)) = directive.split_once('=') else {
            return false;
        };

        TARGETS.contains(&target.trim()) && matches!(level.trim(), "debug" | "trace")
    })
}

/// Opens the file the diagnostics are written to; a sink when no file is configured.
fn gpu_log_file(path: Option<&Path>, limit: u64) -> io::Result<Box<dyn Write + Send>> {
    match path {
        Some(path) => Ok(Box::new(RotatingLogFile::open(path, limit)?)),
        None => Ok(Box::new(io::sink())),
    }
}

/// Opens the diagnostics file for appending, making its directory when that is missing.
///
/// A path whose directory does not exist yet is an easy way to ask for a log that never appears, so
/// the directory is made rather than the open failing; the file itself is created too.
fn open_append(path: &Path) -> io::Result<File> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        create_dir_all(parent)?;
    }

    OpenOptions::new().create(true).append(true).open(path)
}

/// Written at the start of the diagnostics file when the limit dropped output, so that a report of
/// a problem does not read as if the run had just started.
const TRUNCATION_NOTE: &[u8] =
    b"--- the diagnostics file hit its size limit; the output above was dropped ---\n";

/// The diagnostics file, keeping only the most recent `limit` bytes of it.
///
/// The file is opened for appending, so runs add to the same file, and it is followed rather than
/// held on to: a diagnostics file that is deleted or rotated while niri is running would otherwise
/// keep taking writes into a file that nothing can find any more.
struct RotatingLogFile {
    path: PathBuf,
    file: File,
    /// Bytes of the file in use, which is where the next append lands.
    len: u64,
    /// How much of it is kept; 0 means no limit.
    limit: u64,
    /// The file the writer has open, which says whether `path` still names it.
    inode: u64,
}

impl RotatingLogFile {
    fn open(path: &Path, limit: u64) -> io::Result<Self> {
        let file = open_append(path)?;
        let mut len = file.metadata()?.len();

        // Starting over right away beats writing past the limit and only then noticing that the
        // file has been over it since an earlier run.
        if limit != 0 && len >= limit {
            file.set_len(0)?;
            len = 0;
        }

        let inode = file.metadata()?.ino();
        Ok(Self {
            path: path.to_owned(),
            file,
            len,
            limit,
            inode,
        })
    }

    /// Makes room for `incoming` bytes, dropping the oldest output when it does not fit.
    fn make_room(&mut self, incoming: u64) -> io::Result<()> {
        if self.limit == 0 || self.len.saturating_add(incoming) <= self.limit {
            return Ok(());
        }

        self.file.set_len(0)?;
        self.len = 0;

        // A limit too small to hold the note and this write keeps the write.
        if self.limit >= TRUNCATION_NOTE.len() as u64 + incoming {
            self.file.write_all(TRUNCATION_NOTE)?;
            self.len = TRUNCATION_NOTE.len() as u64;
        }

        Ok(())
    }

    /// Keeps the writer on the file that `path` names, opening one when there is none.
    ///
    /// A file that is not the one the writer opened — deleted, rotated, or truncated by something
    /// else — is left alone and replaced by the path's file, so the writes never land in a file
    /// that nothing can find any more.
    fn follow_the_path(&mut self) -> io::Result<()> {
        let unchanged = std::fs::metadata(&self.path)
            .is_ok_and(|metadata| metadata.ino() == self.inode && metadata.len() == self.len);
        if unchanged {
            return Ok(());
        }

        self.file = open_append(&self.path)?;
        let metadata = self.file.metadata()?;
        self.inode = metadata.ino();
        self.len = metadata.len();
        Ok(())
    }
}

impl Write for RotatingLogFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.follow_the_path()?;
        self.make_room(buf.len() as u64)?;

        let written = self.file.write(buf)?;
        self.len += written as u64;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Returns the size limit of the diagnostics file, and why the setting is unusable when it is.
fn gpu_log_limit() -> (u64, Option<String>) {
    let Some(value) = env::var_os(GPU_LOG_MAX_BYTES_ENV) else {
        return (DEFAULT_GPU_LOG_MAX_BYTES, None);
    };
    let value = value.to_string_lossy().into_owned();

    match parse_size(&value) {
        Some(limit) => (limit, None),
        None => (
            DEFAULT_GPU_LOG_MAX_BYTES,
            Some(format!("{value:?} is not a size")),
        ),
    }
}

/// Reads a size: a byte count, optionally with a `K`, `M` or `G` (or `KiB`, `MiB`, `GiB`) suffix.
///
/// `off` is read as no limit, which is what a limit of zero bytes means.
fn parse_size(value: &str) -> Option<u64> {
    let value = value.trim().to_ascii_lowercase();
    if matches!(value.as_str(), "off" | "none" | "unlimited") {
        return Some(0);
    }

    let unit_at = value
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(value.len());
    let (digits, unit) = value.split_at(unit_at);
    let multiplier = match unit.trim() {
        "" | "b" => 1,
        "k" | "kb" | "kib" => 1024,
        "m" | "mb" | "mib" => 1024 * 1024,
        "g" | "gb" | "gib" => 1024 * 1024 * 1024,
        _ => return None,
    };

    digits.parse::<u64>().ok()?.checked_mul(multiplier)
}

/// Formats a byte count for the log, in the largest unit that divides it exactly.
fn format_size(bytes: u64) -> String {
    for (unit, size) in [
        ("GiB", 1024 * 1024 * 1024),
        ("MiB", 1024 * 1024),
        ("KiB", 1024),
    ] {
        if bytes >= size && bytes.is_multiple_of(size) {
            return format!("{} {unit}", bytes / size);
        }
    }

    format!("{bytes} bytes")
}

/// Returns the filter of the standard error log.
///
/// When the diagnostics have a file, they are kept out of the standard error unless the filter
/// names them explicitly.
fn stderr_directives(directives: &str, gpu_log: bool) -> String {
    if !gpu_log {
        return directives.to_owned();
    }

    let mut rv = directives.to_owned();
    for target in [
        "niri::render_helpers::gpu",
        "niri::render_helpers::superres",
    ] {
        let name = target.trim_start_matches("niri::");
        if !directives.contains(name) {
            rv = format!("{rv},{target}=off");
        }
    }

    rv
}

#[cfg(feature = "profile-with-tracy-allocations")]
#[global_allocator]
static GLOBAL: tracy_client::ProfiledAllocator<std::alloc::System> =
    tracy_client::ProfiledAllocator::new(std::alloc::System, 100);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Set backtrace defaults if not set.
    if env::var_os("RUST_BACKTRACE").is_none() {
        env::set_var("RUST_BACKTRACE", "1");
        REMOVE_ENV_RUST_BACKTRACE.store(true, Ordering::Relaxed);
    }
    if env::var_os("RUST_LIB_BACKTRACE").is_none() {
        env::set_var("RUST_LIB_BACKTRACE", "0");
        REMOVE_ENV_RUST_LIB_BACKTRACE.store(true, Ordering::Relaxed);
    }

    init_logging();

    if env::var_os("NOTIFY_SOCKET").is_some() {
        IS_SYSTEMD_SERVICE.store(true, Ordering::Relaxed);

        #[cfg(not(feature = "systemd"))]
        warn!(
            "running as a systemd service, but systemd support is compiled out. \
             Are you sure you did not forget to set `--features systemd`?"
        );
    }

    let cli = Cli::parse();

    if cli.session {
        // If we're starting as a session, assume that the intention is to start on a TTY unless
        // this is a WSL environment. Remove DISPLAY, WAYLAND_DISPLAY or WAYLAND_SOCKET from our
        // environment if they are set, since they will cause the winit backend to be selected
        // instead.
        if env::var_os("WSL_DISTRO_NAME").is_none() {
            if env::var_os("DISPLAY").is_some() {
                warn!("running as a session but DISPLAY is set, removing it");
                env::remove_var("DISPLAY");
            }
            if env::var_os("WAYLAND_DISPLAY").is_some() {
                warn!("running as a session but WAYLAND_DISPLAY is set, removing it");
                env::remove_var("WAYLAND_DISPLAY");
            }
            if env::var_os("WAYLAND_SOCKET").is_some() {
                warn!("running as a session but WAYLAND_SOCKET is set, removing it");
                env::remove_var("WAYLAND_SOCKET");
            }
        }

        // Set the current desktop for xdg-desktop-portal.
        env::set_var("XDG_CURRENT_DESKTOP", "niri");
        // Ensure the session type is set to Wayland for xdg-autostart and Qt apps.
        env::set_var("XDG_SESSION_TYPE", "wayland");
    }

    // Handle subcommands.
    if let Some(subcommand) = cli.subcommand {
        match subcommand {
            Sub::Validate { config } => {
                tracy_client::Client::start();

                config_path(config).load().config?;
                info!("config is valid");
                return Ok(());
            }
            Sub::Msg {
                msg,
                json,
                print_request,
            } => {
                handle_msg(msg, json, print_request)?;
                return Ok(());
            }
            Sub::Panic => cause_panic(),
            Sub::Completions { shell } => {
                match shell {
                    CompletionShell::Nushell => {
                        clap_complete::generate(
                            Nushell,
                            &mut Cli::command(),
                            "niri",
                            &mut io::stdout(),
                        );
                    }
                    other => {
                        let generator = Shell::try_from(other).unwrap();
                        clap_complete::generate(
                            generator,
                            &mut Cli::command(),
                            "niri",
                            &mut io::stdout(),
                        );
                    }
                }
                return Ok(());
            }
        }
    }

    // Needs to be done before starting Tracy, so that it applies to Tracy's threads.
    niri::utils::signals::block_early().unwrap();

    // Avoid starting Tracy for the `niri msg` code path since starting/stopping Tracy is a bit
    // slow.
    tracy_client::Client::start();

    // In on-demand mode, we must shut down Tracy manually to terminate the connection cleanly.
    // Do it from a Drop impl here, so that it runs after the Drop code for all of the state created
    // below, because some of those Drop impls themselves create Tracy spans.
    let _shutdown_tracy = ShutdownTracy;

    info!("starting version {}", &version());

    // Load the config.
    let config_path = config_path(cli.config);
    env::remove_var("NIRI_CONFIG");
    let (config_created_at, config_load_result) = config_path.load_or_create();
    let config_errored = config_load_result.config.is_err();
    let mut config = config_load_result.config.unwrap_or_else(|err| {
        warn!("{err:?}");
        Config::load_default()
    });
    let config_includes = config_load_result.includes;

    let spawn_at_startup = mem::take(&mut config.spawn_at_startup);
    let spawn_sh_at_startup = mem::take(&mut config.spawn_sh_at_startup);
    *CHILD_ENV.write().unwrap() = mem::take(&mut config.environment);

    store_and_increase_nofile_rlimit();

    // Create the main event loop.
    let mut event_loop = EventLoop::<State>::try_new().unwrap();

    // Handle Ctrl+C and other signals.
    niri::utils::signals::listen(&event_loop.handle());

    // Check if we should use the headless backend.
    let headless = env::var("NIRI_BACKEND")
        .map(|v| v.eq_ignore_ascii_case("headless"))
        .unwrap_or(false);

    if headless {
        info!("starting in headless mode (NIRI_BACKEND=headless)");
    }

    // Create the compositor.
    let display = Display::new().unwrap();

    // Increase the buffer size so that it's harder to crash a frozen client with a 1000 Hz mouse.
    set_default_max_buffer_size(&display, 1024 * 1024);

    let mut state = State::new(
        config,
        event_loop.handle(),
        event_loop.get_signal(),
        display,
        headless,
        true,
        cli.session,
    )
    .unwrap();

    // Set WAYLAND_DISPLAY for children.
    let socket_name = state.niri.socket_name.as_deref().unwrap();
    env::set_var("WAYLAND_DISPLAY", socket_name);
    info!(
        "listening on Wayland socket: {}",
        socket_name.to_string_lossy()
    );

    // Set NIRI_SOCKET for children.
    if let Some(ipc) = &state.niri.ipc_server {
        let socket_path = ipc.socket_path.as_deref().unwrap();
        env::set_var(SOCKET_PATH_ENV, socket_path);
        info!("IPC listening on: {}", socket_path.to_string_lossy());
    }

    // Setup xwayland-satellite integration.
    xwayland::satellite::setup(&mut state);
    if let Some(satellite) = &state.niri.satellite {
        let name = satellite.display_name();
        *CHILD_DISPLAY.write().unwrap() = Some(name.to_owned());
        env::set_var("DISPLAY", name);
        info!("listening on X11 socket: {name}");
    } else {
        // Avoid spawning children in the host X11.
        env::remove_var("DISPLAY");
    }

    if cli.session {
        // We're starting as a session. Import our variables.
        import_environment();

        // Inhibit power key handling so we can suspend on it.
        #[cfg(feature = "dbus")]
        if !state.niri.config.borrow().input.disable_power_key_handling {
            if let Err(err) = state.niri.inhibit_power_key() {
                warn!("error inhibiting power key: {err:?}");
            }
        }
    }

    #[cfg(feature = "dbus")]
    dbus::DBusServers::start(&mut state, cli.session);

    #[cfg(feature = "dbus")]
    if cli.session {
        state.niri.a11y.start();
    }

    if env::var_os("NIRI_DISABLE_SYSTEM_MANAGER_NOTIFY").is_none_or(|x| x != "1") {
        // Notify systemd we're ready.
        if let Err(err) = sd_notify::notify(&[NotifyState::Ready]) {
            warn!("error notifying systemd: {err:?}");
        };

        // Send ready notification to the NOTIFY_FD file descriptor.
        if let Err(err) = notify_fd() {
            warn!("error notifying fd: {err:?}");
        }
    }

    watcher::setup(&mut state, &config_path, config_includes);

    // Spawn commands from cli and auto-start.
    spawn(cli.command, None);

    for elem in spawn_at_startup {
        spawn(elem.command, None);
    }
    for elem in spawn_sh_at_startup {
        spawn_sh(elem.command, None);
    }

    // Show the config error notification right away if needed.
    if config_errored {
        state.niri.config_error_notification.show();
        state.ipc_config_loaded(true);
    } else if let Some(path) = config_created_at {
        state.niri.config_error_notification.show_created(path);
    }

    // Run the compositor.
    event_loop
        .run(None, &mut state, |state| state.refresh_and_flush_clients())
        .unwrap();

    Ok(())
}

fn import_environment() {
    let variables = [
        "WAYLAND_DISPLAY",
        "DISPLAY",
        "XDG_CURRENT_DESKTOP",
        "XDG_SESSION_TYPE",
        SOCKET_PATH_ENV,
    ]
    .join(" ");

    let mut init_system_import = String::new();
    if cfg!(feature = "systemd") {
        write!(
            init_system_import,
            "systemctl --user import-environment {variables};"
        )
        .unwrap();
    }
    if cfg!(feature = "dinit") {
        write!(init_system_import, "dinitctl setenv {variables};").unwrap();
    }

    let rv = Command::new("/bin/sh")
        .args([
            "-c",
            &format!(
                "{init_system_import}\
                 hash dbus-update-activation-environment 2>/dev/null && \
                 dbus-update-activation-environment {variables}"
            ),
        ])
        .spawn();
    // Wait for the import process to complete, otherwise services will start too fast without
    // environment variables available.
    match rv {
        Ok(mut child) => match child.wait() {
            Ok(status) => {
                if !status.success() {
                    warn!("import environment shell exited with {status}");
                }
            }
            Err(err) => {
                warn!("error waiting for import environment shell: {err:?}");
            }
        },
        Err(err) => {
            warn!("error spawning shell to import environment: {err:?}");
        }
    }
}

fn env_config_path() -> Option<PathBuf> {
    env::var_os("NIRI_CONFIG")
        .filter(|x| !x.is_empty())
        .map(PathBuf::from)
}

fn default_config_path() -> Option<PathBuf> {
    let Some(dirs) = ProjectDirs::from("", "", "niri") else {
        warn!("error retrieving home directory");
        return None;
    };

    let mut path = dirs.config_dir().to_owned();
    path.push("config.kdl");
    Some(path)
}

fn system_config_path() -> PathBuf {
    PathBuf::from("/etc/niri/config.kdl")
}

fn config_path(cli_path: Option<PathBuf>) -> ConfigPath {
    if let Some(explicit) = cli_path.or_else(env_config_path) {
        return ConfigPath::Explicit(explicit);
    }

    let system_path = system_config_path();

    if let Some(user_path) = default_config_path() {
        ConfigPath::Regular {
            user_path,
            system_path,
        }
    } else {
        // Couldn't find the home directory, or whatever.
        ConfigPath::Explicit(system_path)
    }
}

fn notify_fd() -> anyhow::Result<()> {
    let fd = match env::var("NOTIFY_FD") {
        Ok(notify_fd) => notify_fd.parse()?,
        Err(env::VarError::NotPresent) => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    env::remove_var("NOTIFY_FD");
    let mut notif = unsafe { File::from_raw_fd(fd) };
    notif.write_all(b"READY=1\n")?;
    Ok(())
}

// The wayland-server crate has set_default_max_buffer_size() under a libwayland_1_23 feature, but
// this hard-requires libwayland-server >= 1.23 which is not present on e.g. Ubuntu 24.04. Since
// calling this is an optional enhancement, do it optionally at runtime.
fn set_default_max_buffer_size(display: &Display<State>, size: usize) {
    use std::ffi::c_void;

    unsafe {
        // RTLD_NOLOAD ensures we only get a handle to the libwayland-server that wayland-rs has
        // already loaded into this process, rather than potentially pulling in a different copy.
        let lib = libc::dlopen(
            c"libwayland-server.so.0".as_ptr(),
            libc::RTLD_LAZY | libc::RTLD_NOLOAD,
        );
        if lib.is_null() {
            // It's not really expected that this can happen, maybe if some distro changes the
            // library name?
            warn!("cannot set default max buffer size: libwayland-server.so.0 is not loaded");
            return;
        }

        let sym = libc::dlsym(lib, c"wl_display_set_default_max_buffer_size".as_ptr());
        if sym.is_null() {
            // Expected on libwayland-server < 1.23.
            trace!("wl_display_set_default_max_buffer_size is missing; skipping");
        } else {
            let func: unsafe extern "C" fn(*mut c_void, libc::size_t) = std::mem::transmute(sym);
            let display_ptr = display.handle().backend_handle().display_ptr();
            func(display_ptr.cast(), size);
        }

        libc::dlclose(lib);
    }
}

struct ShutdownTracy;
impl Drop for ShutdownTracy {
    fn drop(&mut self) {
        #[cfg(feature = "profile-with-tracy-ondemand")]
        unsafe {
            tracy_client::sys::___tracy_shutdown_profiler();
        }
    }
}

#[cfg(test)]
mod gpu_log_filter_tests {
    use std::io::Read as _;

    use super::*;

    /// Builds the logging subscriber the compositor starts with, writing to two paths.
    fn subscriber_to_files(
        stderr_path: &Path,
        gpu_path: Option<&Path>,
        stderr_directives: &str,
        gpu_directives: &str,
    ) -> impl tracing::Subscriber + Send + Sync + 'static {
        subscriber(
            Mutex::new(gpu_log_file(Some(stderr_path), 0).unwrap()),
            stderr_directives,
            Mutex::new(gpu_log_file(gpu_path, 0).unwrap()),
            gpu_directives,
        )
    }

    fn read(path: &Path) -> String {
        let mut contents = String::new();
        std::fs::File::open(path)
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        contents
    }

    fn temp_path(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("niri-log-{}-{name}.log", std::process::id()));
        let _ = std::fs::remove_file(&path);
        path
    }

    #[test]
    fn the_diagnostics_file_takes_over_the_targets_it_covers() {
        assert_eq!(
            stderr_directives(DEFAULT_LOG_FILTER, true),
            format!("{DEFAULT_LOG_FILTER},niri::render_helpers::superres=off")
        );

        // A filter that names a target explicitly keeps it, since the user asked for it.
        assert_eq!(
            stderr_directives("niri=info,niri::render_helpers::superres=debug", true),
            "niri=info,niri::render_helpers::superres=debug,niri::render_helpers::gpu=off"
        );

        // Without a file nothing is taken over.
        assert_eq!(
            stderr_directives(DEFAULT_LOG_FILTER, false),
            DEFAULT_LOG_FILTER
        );
    }

    #[test]
    fn diagnostics_are_off_unless_the_filter_or_the_file_asks_for_them() {
        assert!(!diagnostics_asked_for(DEFAULT_LOG_FILTER));
        assert!(diagnostics_asked_for(
            "niri=info,niri::render_helpers::gpu=debug"
        ));
        assert!(!diagnostics_asked_for(
            "niri=debug,niri::render_helpers::gpu=off"
        ));
    }

    #[test]
    fn the_default_filter_does_not_record_the_diagnostics() {
        let stderr = temp_path("default-stderr");
        let subscriber = subscriber_to_files(&stderr, None, DEFAULT_LOG_FILTER, "off");

        tracing::subscriber::with_default(subscriber, || {
            debug!(target: "niri::render_helpers::gpu", "gpu message");
            debug!(target: "niri", "session message");
        });

        let contents = read(&stderr);
        assert!(!contents.contains("gpu message"), "{contents}");
        assert!(contents.contains("session message"), "{contents}");

        std::fs::remove_file(&stderr).unwrap();
    }

    #[test]
    fn the_diagnostics_go_to_their_file_and_not_to_the_standard_error() {
        let stderr = temp_path("split-stderr");
        let gpu = temp_path("split-gpu");
        let subscriber = subscriber_to_files(
            &stderr,
            Some(&gpu),
            &stderr_directives(DEFAULT_LOG_FILTER, true),
            DEFAULT_GPU_LOG_FILTER,
        );

        tracing::subscriber::with_default(subscriber, || {
            debug!(target: "niri::render_helpers::gpu", "gpu message");
            debug!(target: "niri::render_helpers::superres", "upscaling message");
            debug!(target: "niri::elsewhere", "unrelated message");
        });

        let gpu_contents = read(&gpu);
        assert!(gpu_contents.contains("gpu message"), "{gpu_contents}");
        assert!(gpu_contents.contains("upscaling message"), "{gpu_contents}");
        assert!(
            !gpu_contents.contains("unrelated message"),
            "{gpu_contents}"
        );

        let stderr_contents = read(&stderr);
        assert!(
            !stderr_contents.contains("gpu message"),
            "{stderr_contents}"
        );
        assert!(
            !stderr_contents.contains("upscaling message"),
            "{stderr_contents}"
        );
        assert!(
            stderr_contents.contains("unrelated message"),
            "{stderr_contents}"
        );

        std::fs::remove_file(&stderr).unwrap();
        std::fs::remove_file(&gpu).unwrap();
    }

    #[test]
    fn the_diagnostics_file_leaves_the_per_frame_output_to_trace() {
        let stderr = temp_path("trace-stderr");
        let gpu = temp_path("trace-gpu");
        let subscriber = subscriber_to_files(
            &stderr,
            Some(&gpu),
            &stderr_directives(DEFAULT_LOG_FILTER, true),
            DEFAULT_GPU_LOG_FILTER,
        );

        tracing::subscriber::with_default(subscriber, || {
            // What repeats for every pass, window and frame.
            trace!(target: "niri::render_helpers::gpu", "per-pass message");
            // What the diagnostics say about the state once.
            debug!(target: "niri::render_helpers::gpu", "state message");
        });

        let gpu_contents = read(&gpu);
        assert!(!gpu_contents.contains("per-pass message"), "{gpu_contents}");
        assert!(gpu_contents.contains("state message"), "{gpu_contents}");

        std::fs::remove_file(&stderr).unwrap();
        std::fs::remove_file(&gpu).unwrap();
    }

    #[test]
    fn a_missing_directory_is_created_for_the_diagnostics_file() {
        let mut path = std::env::temp_dir().join(format!("niri-log-{}-nested", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path.push("gpu.log");

        gpu_log_file(Some(&path), 0).unwrap();
        assert!(path.exists(), "{}", path.display());

        std::fs::remove_dir_all(path.parent().unwrap()).unwrap();
    }

    #[test]
    fn the_size_setting_reads_bytes_with_an_optional_unit() {
        assert_eq!(parse_size("67108864"), Some(64 * 1024 * 1024));
        assert_eq!(parse_size("64M"), Some(64 * 1024 * 1024));
        assert_eq!(parse_size("64 MiB"), Some(64 * 1024 * 1024));
        assert_eq!(parse_size("512k"), Some(512 * 1024));
        assert_eq!(parse_size("2G"), Some(2 * 1024 * 1024 * 1024));
        assert_eq!(parse_size("0"), Some(0));
        assert_eq!(parse_size("off"), Some(0));

        assert_eq!(parse_size(""), None);
        assert_eq!(parse_size("lots"), None);
        assert_eq!(parse_size("64 petabytes"), None);

        assert_eq!(format_size(64 * 1024 * 1024), "64 MiB");
        assert_eq!(format_size(1024), "1 KiB");
        assert_eq!(format_size(3000), "3000 bytes");
    }

    #[test]
    fn the_diagnostics_file_keeps_the_newest_output_within_its_limit() {
        let path = temp_path("limited");
        const LIMIT: u64 = 4096;
        let mut writer = gpu_log_file(Some(&path), LIMIT).unwrap();

        writer
            .write_all(b"the old output the limit drops\n")
            .unwrap();
        writer.write_all(&[b'x'; LIMIT as usize]).unwrap();
        writer
            .write_all(b"the new output the limit keeps\n")
            .unwrap();
        drop(writer);

        let contents = read(&path);
        assert!(!contents.contains("the old output"), "{contents}");
        assert!(contents.contains("the new output"), "{contents}");
        assert!(contents.len() as u64 <= LIMIT, "{}", contents.len());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_diagnostics_file_deleted_while_running_comes_back() {
        let path = temp_path("deleted");
        let mut writer = gpu_log_file(Some(&path), 0).unwrap();

        writer.write_all(b"before the deletion\n").unwrap();
        std::fs::remove_file(&path).unwrap();
        writer.write_all(b"after the deletion\n").unwrap();
        drop(writer);

        assert!(path.exists(), "{}", path.display());
        let contents = read(&path);
        assert!(!contents.contains("before the deletion"), "{contents}");
        assert!(contents.contains("after the deletion"), "{contents}");

        std::fs::remove_file(&path).unwrap();
    }
}
