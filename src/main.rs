mod mining;
mod output;
mod pattern;
mod plc;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use mining::{MiningBackend, MiningConfig};
use output::{Styles, fmt_count, fmt_duration, human, print_match, register_did};
use pattern::{difficulty, validate_pattern};

/// Mine vanity did:plc identifiers
#[derive(Parser)]
#[command(name = "plcpick", version)]
struct Cli {
    /// Pattern to match (e.g. grug*, *grug, did:plc:grug*)
    /// Only base32 chars (a-z, 2-7) and * wildcards.
    pattern: String,

    /// Handle for the DID document (e.g. user.bsky.social)
    #[arg(long, required_unless_present = "placeholder", requires = "pds")]
    handle: Option<String>,

    /// PDS endpoint URL (e.g. https://bsky.social)
    #[arg(long, required_unless_present = "placeholder", requires = "handle")]
    pds: Option<String>,

    /// Use placeholder values instead of real handle/PDS
    #[arg(long, conflicts_with_all = ["handle", "pds"])]
    placeholder: bool,

    /// Keep mining after first match (Ctrl+C to stop)
    #[arg(long)]
    keep_going: bool,

    /// Submit genesis operation to plc.directory on match
    #[arg(long)]
    register: bool,

    /// Number of mining threads (default: all CPU cores)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Mining backend to use (auto, cpu, cuda, wgpu)
    #[arg(long, default_value = "auto")]
    backend: String,
}

/// Auto-detect the best available backend: CUDA > Vulkan > CPU.
fn select_backend(threads: usize) -> Box<dyn MiningBackend> {
    #[cfg(feature = "cuda")]
    {
        if cudarc::driver::CudaDevice::new(0).is_ok() {
            return Box::new(mining::cuda::CudaBackend { device_id: 0 });
        }
    }

    #[cfg(feature = "wgpu")]
    {
        // wgpu auto-detects Vulkan/Metal/DX12 — just check if any adapter exists
        let instance = wgpu::Instance::default();
        if pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })).is_ok() {
            return Box::new(mining::wgpu_backend::WgpuBackend { device_index: 0 });
        }
    }

    Box::new(mining::cpu::CpuBackend { threads })
}

fn main() {
    let cli = Cli::parse();

    let pattern = match validate_pattern(&cli.pattern) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let handle = cli.handle.as_deref().unwrap_or("handle.invalid");
    let pds = cli.pds.as_deref().unwrap_or("https://pds.invalid");
    let threads = cli.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let keep_going = cli.keep_going;
    let s = Styles::new();

    // select backend
    let backend: Box<dyn MiningBackend> = match cli.backend.as_str() {
        "auto" => select_backend(threads),
        "cpu" => Box::new(mining::cpu::CpuBackend { threads }),
        #[cfg(feature = "cuda")]
        "cuda" => Box::new(mining::cuda::CudaBackend { device_id: 0 }),
        #[cfg(feature = "wgpu")]
        "wgpu" => Box::new(mining::wgpu_backend::WgpuBackend { device_index: 0 }),
        other => {
            let mut available = vec!["auto", "cpu"];
            if cfg!(feature = "cuda") { available.push("cuda"); }
            if cfg!(feature = "wgpu") { available.push("wgpu"); }
            eprintln!("error: unknown backend '{other}'. available: {}", available.join(", "));
            std::process::exit(1);
        }
    };

    // header
    let diff = difficulty(&pattern);
    println!();
    println!(
        "  {} {}",
        s.dim.apply_to("plcpick"),
        s.dim.apply_to(env!("CARGO_PKG_VERSION")),
    );
    println!(
        "  {} {}",
        s.dim.apply_to("pattern   "),
        s.cyan.apply_to(&pattern),
    );
    println!(
        "  {} {}",
        s.dim.apply_to("difficulty"),
        s.dim.apply_to(format!("~{} attempts", human(diff))),
    );
    println!(
        "  {} {}",
        s.dim.apply_to("backend   "),
        backend.name(),
    );
    println!(
        "  {} {}",
        s.dim.apply_to("threads   "),
        threads,
    );
    println!(
        "  {} {}",
        s.dim.apply_to("handle    "),
        handle,
    );
    println!(
        "  {} {}",
        s.dim.apply_to("pds       "),
        pds,
    );
    println!();

    // progress spinner
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("  {spinner:.cyan} {msg}")
            .unwrap(),
    );

    let stop = AtomicBool::new(false);
    let total = AtomicU64::new(0);
    let start = Instant::now();
    let (tx, rx) = mpsc::channel::<mining::Match>();

    let config = MiningConfig {
        pattern: pattern.as_bytes().to_vec(),
        handle: handle.to_string(),
        pds: pds.to_string(),
        keep_going,
    };

    // run backend in a separate thread so main can handle output
    let backend_thread = {
        let stop_ptr = &stop as *const AtomicBool as usize;
        let total_ptr = &total as *const AtomicU64 as usize;
        std::thread::spawn(move || {
            // SAFETY: stop and total live on main's stack, outlive this thread
            let stop = unsafe { &*(stop_ptr as *const AtomicBool) };
            let total = unsafe { &*(total_ptr as *const AtomicU64) };
            if let Err(e) = backend.run(&config, stop, &total, tx) {
                eprintln!("error: backend failed: {e}");
            }
        })
    };

    // main loop: progress updates + match output
    loop {
        match rx.try_recv() {
            Ok(m) => {
                pb.suspend(|| {
                    print_match(&m, &s);

                    if cli.register {
                        println!();
                        eprint!("  registering with plc.directory... ");
                        match register_did(&m.did, &m.op) {
                            Ok(()) => eprintln!("{}", s.green.apply_to("done")),
                            Err(e) => eprintln!("{}", s.red.apply_to(format!("failed: {e}"))),
                        }
                    }

                    println!();
                });

                if !keep_going {
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => break,
            Err(mpsc::TryRecvError::Empty) => {
                let n = total.load(Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64().max(0.001);
                let rate = n as f64 / elapsed;
                let eta = if rate > 0.0 && n > 0 {
                    let remaining = diff.saturating_sub(n) as f64 / rate;
                    if remaining <= 0.0 {
                        "any moment".to_string()
                    } else {
                        format!("~{}", fmt_duration(remaining))
                    }
                } else {
                    "...".to_string()
                };
                pb.set_message(format!(
                    "mining...  {}  |  {:.0}/s  |  {}  |  ETA {}",
                    fmt_count(n),
                    rate,
                    fmt_duration(elapsed),
                    eta,
                ));
                pb.tick();
                std::thread::sleep(Duration::from_millis(80));
            }
        }
    }

    pb.finish_and_clear();
    let _ = backend_thread.join();
}
