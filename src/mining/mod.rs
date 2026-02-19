pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::mpsc;
use std::time::Duration;

use crate::plc::PlcOperation;

pub struct MiningConfig {
    pub pattern: Vec<u8>,
    pub handle: String,
    pub pds: String,
    pub keep_going: bool,
}

pub struct Match {
    pub did: String,
    pub key_hex: String,
    pub op: PlcOperation,
    pub attempts: u64,
    pub elapsed: Duration,
}

pub trait MiningBackend: Send + Sync {
    fn name(&self) -> &str;

    /// Start mining. Sends matches via tx, bumps total counter,
    /// stops when stop flag is set.
    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>>;
}
