use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use k256::ecdsa::SigningKey;

use super::{Match, MiningBackend, MiningConfig};
use crate::pattern::glob_match;
use crate::plc::{build_signed_op, did_suffix};

pub struct CpuBackend {
    pub threads: usize,
}

impl MiningBackend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn run(
        &self,
        config: &MiningConfig,
        stop: &AtomicBool,
        total: &AtomicU64,
        tx: mpsc::Sender<Match>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let start = Instant::now();
        let pat: Arc<[u8]> = config.pattern.clone().into();
        let handle: Arc<str> = config.handle.as_str().into();
        let pds: Arc<str> = config.pds.as_str().into();
        let keep_going = config.keep_going;

        let mut handles = Vec::with_capacity(self.threads);

        for _ in 0..self.threads {
            let tx = tx.clone();
            let stop = stop as *const AtomicBool as usize;
            let total = total as *const AtomicU64 as usize;
            let pat = Arc::clone(&pat);
            let handle = Arc::clone(&handle);
            let pds = Arc::clone(&pds);

            handles.push(std::thread::spawn(move || {
                // SAFETY: stop and total outlive the thread (caller guarantees)
                let stop = unsafe { &*(stop as *const AtomicBool) };
                let total = unsafe { &*(total as *const AtomicU64) };

                let mut rng = rand::thread_rng();
                let mut local: u64 = 0;

                loop {
                    if stop.load(Ordering::Relaxed) && !keep_going {
                        break;
                    }

                    let key = SigningKey::random(&mut rng);
                    let op = build_signed_op(&key, &handle, &pds);
                    let suffix = did_suffix(&op);

                    local += 1;
                    if local % 512 == 0 {
                        total.fetch_add(512, Ordering::Relaxed);
                    }

                    if glob_match(&pat, suffix.as_bytes()) {
                        total.fetch_add(local % 512, Ordering::Relaxed);
                        local = 0;

                        let m = Match {
                            did: format!("did:plc:{suffix}"),
                            key_hex: data_encoding::HEXLOWER.encode(&key.to_bytes()),
                            op,
                            attempts: total.load(Ordering::Relaxed),
                            elapsed: start.elapsed(),
                        };

                        if tx.send(m).is_err() {
                            break;
                        }

                        if !keep_going {
                            stop.store(true, Ordering::Relaxed);
                            break;
                        }
                    }
                }
            }));
        }

        // Wait for all threads to finish
        for h in handles {
            let _ = h.join();
        }

        Ok(())
    }
}
