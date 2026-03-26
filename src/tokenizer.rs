//! 軽量 BPE トークナイザー — HuggingFace `tokenizer.json` 互換。
//!
//! Qwen3.5 の BPE トークナイザーを外部依存なしで実装。
//! `tokenizer.json` から vocab + merges を読み込み、encode/decode を行う。

use serde::Deserialize;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// BPE トークナイザー。
pub struct BpeTokenizer {
    /// トークン文字列 → ID。
    token_to_id: HashMap<String, u32>,
    /// ID → トークン文字列。
    id_to_token: Vec<String>,
    /// BPE マージルール: (left, right) → マージ優先度 (小さいほど優先)。
    merges: HashMap<(String, String), usize>,
    /// EOS トークン ID。
    pub eos_token_id: u32,
    /// `<|im_start|>` トークン ID。
    pub im_start_id: u32,
    /// `<|im_end|>` トークン ID。
    pub im_end_id: u32,
}

/// tokenizer.json のトップレベル構造 (必要フィールドのみ)。
#[derive(Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

/// tokenizer.json の model セクション。
#[derive(Deserialize)]
struct TokenizerModel {
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

/// tokenizer.json の added_tokens エントリ。
#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

impl BpeTokenizer {
    /// `tokenizer.json` ファイルからトークナイザーを構築。
    ///
    /// # Errors
    ///
    /// ファイル読み込みまたは JSON パースに失敗した場合。
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        Self::from_json(&data)
    }

    /// JSON 文字列からトークナイザーを構築。
    ///
    /// # Errors
    ///
    /// JSON パースに失敗した場合。
    pub fn from_json(json_str: &str) -> io::Result<Self> {
        let tj: TokenizerJson = serde_json::from_str(json_str)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // vocab: token → id
        let mut token_to_id = tj.model.vocab;

        // added tokens を追加
        for at in &tj.added_tokens {
            token_to_id.insert(at.content.clone(), at.id);
        }

        // id → token (逆引き)
        let max_id = token_to_id.values().copied().max().unwrap_or(0) as usize;
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (tok, &id) in &token_to_id {
            let idx = id as usize;
            if idx < id_to_token.len() {
                id_to_token[idx].clone_from(tok);
            }
        }

        // merges: "Ġ Ġ" → (left="Ġ", right="Ġ") with priority
        let mut merges = HashMap::with_capacity(tj.model.merges.len());
        for (priority, merge_str) in tj.model.merges.iter().enumerate() {
            if let Some((left, right)) = merge_str.split_once(' ') {
                merges.insert((left.to_string(), right.to_string()), priority);
            }
        }

        // special token IDs
        let eos_token_id = token_to_id.get("<|im_end|>").copied().unwrap_or(248_046);
        let im_start_id = token_to_id.get("<|im_start|>").copied().unwrap_or(248_045);
        let im_end_id = eos_token_id;

        Ok(Self {
            token_to_id,
            id_to_token,
            merges,
            eos_token_id,
            im_start_id,
            im_end_id,
        })
    }

    /// Vocab サイズを返す。
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// テキストをトークン ID 列にエンコード。
    ///
    /// Qwen3.5 の pre-tokenizer は GPT-2 スタイル:
    /// スペースを `Ġ` (U+0120) に変換し、UTF-8 バイトを対応文字にマッピング。
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Pre-tokenize: UTF-8 バイト → GPT-2 バイトレベル文字
        let byte_str = text_to_byte_tokens(text);

        // 単語分割 (簡易: 空白・句読点で分割)
        let words = pre_tokenize_words(&byte_str);

        let mut all_ids = Vec::new();
        for word in &words {
            // 各文字を初期トークンとして分割
            let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

            // BPE マージを繰り返し適用
            loop {
                if symbols.len() < 2 {
                    break;
                }

                // 最優先マージペアを見つける
                let mut best_pair = None;
                let mut best_priority = usize::MAX;

                for i in 0..symbols.len() - 1 {
                    let pair = (symbols[i].clone(), symbols[i + 1].clone());
                    if let Some(&p) = self.merges.get(&pair) {
                        if p < best_priority {
                            best_priority = p;
                            best_pair = Some(i);
                        }
                    }
                }

                let Some(idx) = best_pair else {
                    break;
                };

                // マージ実行
                let merged = format!("{}{}", symbols[idx], symbols[idx + 1]);
                symbols[idx] = merged;
                symbols.remove(idx + 1);
            }

            // トークン文字列 → ID
            for sym in &symbols {
                if let Some(&id) = self.token_to_id.get(sym) {
                    all_ids.push(id);
                }
                // 未知トークンはスキップ (byte_fallback=false)
            }
        }

        all_ids
    }

    /// トークン ID 列をテキストにデコード。
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut byte_str = String::new();
        for &id in ids {
            let idx = id as usize;
            if idx < self.id_to_token.len() {
                byte_str.push_str(&self.id_to_token[idx]);
            }
        }

        // GPT-2 バイトレベル文字 → UTF-8 バイト
        byte_tokens_to_text(&byte_str)
    }

    /// Chat テンプレートでプロンプトをフォーマット。
    ///
    /// Qwen3.5 の ChatML 形式:
    /// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n`
    #[must_use]
    pub fn format_chat(&self, system: &str, user: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // system
        if !system.is_empty() {
            ids.push(self.im_start_id);
            ids.extend(self.encode("system\n"));
            ids.extend(self.encode(system));
            ids.push(self.im_end_id);
            ids.extend(self.encode("\n"));
        }

        // user
        ids.push(self.im_start_id);
        ids.extend(self.encode("user\n"));
        ids.extend(self.encode(user));
        ids.push(self.im_end_id);
        ids.extend(self.encode("\n"));

        // assistant prefix
        ids.push(self.im_start_id);
        ids.extend(self.encode("assistant\n"));

        ids
    }

    /// 特定のトークン文字列を ID に変換。
    #[must_use]
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

// ── GPT-2 バイトレベルエンコーディング ────────────────────────────────────

/// GPT-2 バイトレベル: バイト値 → Unicode 文字のマッピングテーブル。
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    // 印字可能 ASCII + Latin-1 Supplement の一部はそのまま
    // '!' (33) ~ '~' (126), '¡' (161) ~ '¬' (172), '®' (174) ~ 'ÿ' (255)
    let mut n = 0u32;
    for b in 0u16..=255 {
        let c = b as u8;
        let is_printable =
            (33..=126).contains(&c) || (161..=172).contains(&c) || (174..=255).contains(&c);
        if is_printable {
            table[c as usize] = char::from(c);
        } else {
            // 非印字可能バイトは U+0100 以降にマッピング
            table[c as usize] = char::from_u32(256 + n).unwrap_or('?');
            n += 1;
        }
    }
    table
}

/// Unicode 文字 → バイト値の逆マッピング。
fn unicode_to_byte() -> HashMap<char, u8> {
    let fwd = byte_to_unicode();
    let mut rev = HashMap::with_capacity(256);
    for (b, &c) in fwd.iter().enumerate() {
        rev.insert(c, b as u8);
    }
    rev
}

/// テキスト → GPT-2 バイトレベル文字列。
fn text_to_byte_tokens(text: &str) -> String {
    let table = byte_to_unicode();
    let mut out = String::with_capacity(text.len() * 2);
    for &b in text.as_bytes() {
        out.push(table[b as usize]);
    }
    out
}

/// GPT-2 バイトレベル文字列 → テキスト。
fn byte_tokens_to_text(s: &str) -> String {
    let rev = unicode_to_byte();
    let bytes: Vec<u8> = s.chars().filter_map(|c| rev.get(&c).copied()).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// 簡易 pre-tokenizer: 空白・記号境界で単語分割。
///
/// GPT-2/Qwen スタイル: スペースは前のトークンに付かず、次のトークンの先頭に `Ġ` として付く。
fn pre_tokenize_words(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut words = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        if c == 'Ġ' || c == ' ' {
            // スペース境界: 現在の単語を確定し、新しい単語を Ġ で開始
            if !current.is_empty() {
                words.push(current);
                current = String::new();
            }
            current.push('Ġ');
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        words.push(current);
    }

    words
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_tokenizer_json() -> String {
        r#"{
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": {
                    "h": 0, "e": 1, "l": 2, "o": 3, "Ġ": 4,
                    "he": 5, "ll": 6, "Ġw": 7, "or": 8,
                    "hel": 9, "llo": 10, "Ġwo": 11, "rld": 12,
                    "hello": 13, "Ġworld": 14
                },
                "merges": [
                    "h e",
                    "l l",
                    "o r",
                    "Ġ w",
                    "he l",
                    "ll o",
                    "Ġw o",
                    "or l",
                    "hel lo",
                    "Ġwo rl",
                    "Ġworl d"
                ],
                "byte_fallback": false
            },
            "added_tokens": [
                {"id": 100, "content": "<|im_start|>", "special": true},
                {"id": 101, "content": "<|im_end|>", "special": true}
            ]
        }"#
        .to_string()
    }

    #[test]
    fn test_load_tokenizer() {
        let tok = BpeTokenizer::from_json(&mini_tokenizer_json()).unwrap();
        assert!(tok.vocab_size() > 10);
        assert_eq!(tok.im_start_id, 100);
        assert_eq!(tok.im_end_id, 101);
    }

    #[test]
    fn test_byte_to_unicode_roundtrip() {
        let fwd = byte_to_unicode();
        let rev = unicode_to_byte();
        for b in 0u8..=255 {
            let c = fwd[b as usize];
            assert_eq!(rev[&c], b);
        }
    }

    #[test]
    fn test_text_byte_roundtrip() {
        let text = "Hello, world! 日本語テスト";
        let encoded = text_to_byte_tokens(text);
        let decoded = byte_tokens_to_text(&encoded);
        assert_eq!(text, decoded);
    }

    #[test]
    fn test_pre_tokenize_words() {
        let text = text_to_byte_tokens("hello world");
        let words = pre_tokenize_words(&text);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0], "hello");
        assert!(words[1].starts_with('Ġ')); // "Ġworld"
    }

    #[test]
    fn test_encode_decode_basic() {
        let tok = BpeTokenizer::from_json(&mini_tokenizer_json()).unwrap();
        // "hello" → BPE should merge to single token
        let ids = tok.encode("hello");
        assert!(!ids.is_empty());
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_encode_empty() {
        let tok = BpeTokenizer::from_json(&mini_tokenizer_json()).unwrap();
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_format_chat() {
        let tok = BpeTokenizer::from_json(&mini_tokenizer_json()).unwrap();
        let ids = tok.format_chat("", "hello");
        // Should contain im_start, "user\n", "hello", im_end, "\n", im_start, "assistant\n"
        assert!(ids.contains(&100)); // im_start
        assert!(ids.contains(&101)); // im_end
    }

    #[test]
    fn test_token_id_lookup() {
        let tok = BpeTokenizer::from_json(&mini_tokenizer_json()).unwrap();
        assert_eq!(tok.token_id("<|im_start|>"), Some(100));
        assert_eq!(tok.token_id("<|im_end|>"), Some(101));
        assert_eq!(tok.token_id("nonexistent"), None);
    }
}
