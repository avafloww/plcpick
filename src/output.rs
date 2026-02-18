use console::Style;

use crate::plc::PlcOperation;

pub struct Styles {
    pub dim: Style,
    pub green: Style,
    pub yellow: Style,
    pub cyan: Style,
    pub red: Style,
}

impl Styles {
    pub fn new() -> Self {
        Self {
            dim: Style::new().dim(),
            green: Style::new().green().bold(),
            yellow: Style::new().yellow(),
            cyan: Style::new().cyan(),
            red: Style::new().red(),
        }
    }
}

pub fn print_match(m: &crate::mining::Match, s: &Styles) {
    let rate = m.attempts as f64 / m.elapsed.as_secs_f64().max(0.001);

    println!();
    println!(
        "  {} {}",
        s.green.apply_to("found"),
        s.green.apply_to(&m.did),
    );
    println!(
        "  {}",
        s.dim.apply_to(format!(
            "{} attempts, {:.1}s, {:.0}/s",
            fmt_count(m.attempts),
            m.elapsed.as_secs_f64(),
            rate,
        )),
    );
    println!();
    println!("  {}", s.yellow.apply_to("private key (keep this safe!):"));
    println!("  {}", m.key_hex);
    println!();
    println!("  {}", s.dim.apply_to("genesis operation:"));
    let json = serde_json::to_string_pretty(&m.op).unwrap();
    for line in json.lines() {
        println!("  {}", line);
    }
}

pub fn fmt_count(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

pub fn fmt_duration(secs: f64) -> String {
    if secs < 60.0 {
        format!("{:.0}s", secs)
    } else if secs < 3600.0 {
        let m = (secs / 60.0) as u64;
        let s = (secs % 60.0) as u64;
        format!("{m}m {s:02}s")
    } else {
        let h = (secs / 3600.0) as u64;
        let m = ((secs % 3600.0) / 60.0) as u64;
        format!("{h}h {m:02}m")
    }
}

pub fn human(n: u64) -> String {
    match n {
        n if n >= 1_000_000_000_000 => format!("{:.1}T", n as f64 / 1e12),
        n if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        n if n >= 1_000 => format!("{:.1}K", n as f64 / 1e3),
        n => format!("{n}"),
    }
}

pub fn register_did(did: &str, op: &PlcOperation) -> Result<(), String> {
    let body = serde_json::to_string(op).map_err(|e| e.to_string())?;
    let url = format!("https://plc.directory/{did}");

    match ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string(&body)
    {
        Ok(_) => Ok(()),
        Err(ureq::Error::Status(code, resp)) => {
            let msg = resp.into_string().unwrap_or_default();
            Err(format!("HTTP {code}: {msg}"))
        }
        Err(ureq::Error::Transport(e)) => Err(format!("{e}")),
    }
}
