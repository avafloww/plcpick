# plcpick

A vanity `did:plc` identifier miner for [AT Protocol](https://atproto.com/). Find a DID matching any glob pattern — `cool*`, `*vibe`, `a*z`, whatever you like.

## How it works

Every AT Protocol account has a `did:plc` identifier containing a 24-character base32 string derived from the SHA256 hash of a signed CBOR genesis operation. plcpick brute-forces secp256k1 keys until one produces a DID matching your pattern.

The full pipeline per attempt: generate key → build signed PLC operation → CBOR-encode → SHA256 → truncate → base32 → pattern match.

## Installation

```bash
# CPU-only (default)
cargo install --path .

# With CUDA GPU acceleration (requires CUDA toolkit / nvcc)
cargo install --path . --features cuda
```

### Requirements

- Rust (2024 edition, nightly or recent stable)
- For CUDA: NVIDIA GPU (Turing or newer, sm_75+) and CUDA toolkit with `nvcc`

## Usage

```bash
# Mine a DID starting with "cool"
plcpick 'cool*' --handle yourname.bsky.social --pds https://bsky.social

# Mine a DID ending with "vibe"
plcpick '*vibe' --handle yourname.bsky.social --pds https://bsky.social

# Wildcard in the middle
plcpick 'a*z' --handle yourname.bsky.social --pds https://bsky.social

# Use placeholder values (for testing, not registerable)
plcpick 'cool*' --placeholder

# Keep mining after the first match
plcpick 'cool*' --placeholder --keep-going

# Auto-register the DID with plc.directory on match
plcpick 'cool*' --handle yourname.bsky.social --pds https://bsky.social --register

# Use CUDA backend (requires --features cuda)
plcpick 'cool*' --placeholder --backend cuda

# Control CPU thread count
plcpick 'cool*' --placeholder --threads 4
```

Patterns use base32 characters only (`a-z`, `2-7`) and `*` wildcards. The `did:plc:` prefix is optional and stripped automatically.

## Feature Flags

| Flag | Description |
|------|-------------|
| *(default)* | CPU-only backend. Pure Rust, no system dependencies. |
| `cuda` | Adds CUDA GPU backend. Runs the entire mining pipeline on-GPU using a CBOR template patching strategy for maximum throughput. Requires `nvcc` at build time. |

## License

[MIT](LICENSE)
