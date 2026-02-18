use console::Style;

use crate::plc::PlcOperation;

pub struct Styles {
    pub dim: Style,
    pub label: Style,
    pub green: Style,
    pub yellow: Style,
    pub cyan: Style,
    pub red: Style,
    pub magenta: Style,
    pub value: Style,
}

impl Styles {
    pub fn new() -> Self {
        Self {
            dim: Style::new().dim(),
            label: Style::new().color256(245),         // medium gray — readable but recessive
            green: Style::new().color256(114).bold(),   // bright mint green
            yellow: Style::new().color256(220).bold(),  // vivid amber
            cyan: Style::new().color256(81),            // bright sky blue
            red: Style::new().color256(203).bold(),     // bright coral red
            magenta: Style::new().color256(176),        // soft pink/magenta
            value: Style::new().white().bold(),         // bright white for values
        }
    }
}

pub fn print_match(m: &crate::mining::Match, s: &Styles) {
    let rate = m.attempts as f64 / m.elapsed.as_secs_f64().max(0.001);

    println!();
    println!(
        "  {} {}",
        s.green.apply_to("✓"),
        s.green.apply_to(&m.did),
    );
    println!(
        "    {}",
        s.dim.apply_to(format!(
            "{} attempts  ·  {}  ·  {}/s",
            fmt_count(m.attempts),
            fmt_duration(m.elapsed.as_secs_f64()),
            fmt_rate(rate),
        )),
    );
    println!();
    println!(
        "  {} {}",
        s.yellow.apply_to("⚠"),
        s.yellow.apply_to("private key (keep this safe!)"),
    );
    println!("    {}", s.value.apply_to(&m.key_hex));
    println!();
    println!(
        "  {} {}",
        s.label.apply_to("▸"),
        s.label.apply_to("genesis operation"),
    );
    let json = serde_json::to_string_pretty(&m.op).unwrap();
    for line in json.lines() {
        println!("    {}", s.dim.apply_to(line));
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

pub fn fmt_rate(rate: f64) -> String {
    let n = rate as u64;
    match n {
        n if n >= 1_000_000_000 => format!("{:.1}B", rate / 1e9),
        n if n >= 1_000_000 => format!("{:.1}M", rate / 1e6),
        n if n >= 1_000 => format!("{:.1}K", rate / 1e3),
        _ => format!("{:.0}", rate),
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
