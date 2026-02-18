pub fn validate_pattern(input: &str) -> Result<String, String> {
    let pat = input
        .strip_prefix("did:plc:")
        .unwrap_or(input)
        .to_ascii_lowercase();

    if pat.is_empty() {
        return Err("pattern is empty".into());
    }

    if pat.bytes().all(|b| b == b'*') {
        return Err("pattern is all wildcards — matches everything".into());
    }

    for b in pat.bytes() {
        match b {
            b'*' | b'a'..=b'z' | b'2'..=b'7' => {}
            _ => {
                return Err(format!(
                    "invalid char '{}' — only a-z, 2-7, * allowed (base32)",
                    b as char,
                ));
            }
        }
    }

    Ok(pat)
}

pub fn glob_match(pattern: &[u8], text: &[u8]) -> bool {
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star = usize::MAX;
    let mut star_t = 0usize;

    while ti < text.len() {
        if pi < pattern.len() && pattern[pi] == text[ti] {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == b'*' {
            star = pi;
            star_t = ti;
            pi += 1;
        } else if star != usize::MAX {
            pi = star + 1;
            star_t += 1;
            ti = star_t;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == b'*' {
        pi += 1;
    }

    pi == pattern.len()
}

pub fn difficulty(pattern: &str) -> u64 {
    let fixed = pattern.bytes().filter(|&b| b != b'*').count() as u32;
    32u64.saturating_pow(fixed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_strips_did_prefix() {
        assert_eq!(validate_pattern("did:plc:abc*").unwrap(), "abc*");
    }

    #[test]
    fn validate_lowercases() {
        assert_eq!(validate_pattern("ABC*").unwrap(), "abc*");
    }

    #[test]
    fn validate_rejects_empty() {
        assert!(validate_pattern("").is_err());
    }

    #[test]
    fn validate_rejects_all_wildcards() {
        assert!(validate_pattern("***").is_err());
    }

    #[test]
    fn validate_rejects_invalid_chars() {
        assert!(validate_pattern("abc8").is_err());
        assert!(validate_pattern("abc!").is_err());
        assert!(validate_pattern("abc1").is_err());
        assert!(validate_pattern("abc0").is_err());
    }

    #[test]
    fn validate_accepts_base32() {
        assert!(validate_pattern("abcdefg234567").is_ok());
    }

    #[test]
    fn glob_exact_match() {
        assert!(glob_match(b"abc", b"abc"));
        assert!(!glob_match(b"abc", b"abd"));
    }

    #[test]
    fn glob_prefix_wildcard() {
        assert!(glob_match(b"abc*", b"abcdef"));
        assert!(glob_match(b"abc*", b"abc"));
        assert!(!glob_match(b"abc*", b"ab"));
    }

    #[test]
    fn glob_suffix_wildcard() {
        assert!(glob_match(b"*abc", b"xyzabc"));
        assert!(glob_match(b"*abc", b"abc"));
        assert!(!glob_match(b"*abc", b"abx"));
    }

    #[test]
    fn glob_middle_wildcard() {
        assert!(glob_match(b"a*c", b"abc"));
        assert!(glob_match(b"a*c", b"aXXXc"));
        assert!(!glob_match(b"a*c", b"aXXXd"));
    }

    #[test]
    fn glob_multiple_wildcards() {
        assert!(glob_match(b"*a*b*", b"xaxbx"));
        assert!(glob_match(b"*a*b*", b"ab"));
    }

    #[test]
    fn glob_all_wildcard() {
        assert!(glob_match(b"*", b"anything"));
        assert!(glob_match(b"*", b""));
    }

    #[test]
    fn difficulty_calculation() {
        assert_eq!(difficulty("*"), 1);
        assert_eq!(difficulty("a*"), 32);
        assert_eq!(difficulty("ab*"), 1024);
    }
}
