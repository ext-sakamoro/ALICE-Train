//! PyTorch `.pth` ファイルリーダー。
//!
//! `torch.save(state_dict, "model.pth")` で保存されたファイルを読み込む。
//!
//! # フォーマット
//!
//! PyTorch 1.6+ の `.pth` は ZIP アーカイブ。内部構造:
//!
//! ```text
//! archive/
//!   data.pkl          — pickle プロトコル 2/4。テンソルのメタデータ (dtype, shape, storage key)
//!   data/
//!     0               — storage key 0 のバイナリデータ
//!     1               — storage key 1 のバイナリデータ
//!     ...
//! ```
//!
//! pickle の完全解析はせず、メタデータを独自パーサで抽出する。
//! 対応 dtype: `float32`, `bfloat16`, `float16`, `float64`, `int32`, `int64`, `int8`, `uint8`.
//!
//! # 制限事項
//!
//! - ネストしたサブモジュール名（`layer.0.weight` 等）は `.` 区切りで保持する
//! - `torch.save(tensor)` 単体保存は未対応（`state_dict` 形式のみ）
//! - Python オブジェクト（`nn.Module` 全体保存）は未対応
//!
//! # 使用例
//!
//! ```rust,no_run
//! use alice_train::pth_loader::PthLoader;
//!
//! let loader = PthLoader::open("model.pth").unwrap();
//! println!("テンソル数: {}", loader.tensor_count());
//! for name in loader.tensor_names() {
//!     if let Some(tensor) = loader.get_tensor_f32(&name) {
//!         println!("{}: {} 要素", name, tensor.len());
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::io::{self, Read};
use std::path::Path;

/// テンソルのデータ型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PthDtype {
    /// 32-bit 浮動小数点。
    Float32,
    /// Brain float 16-bit。
    BFloat16,
    /// 16-bit 浮動小数点。
    Float16,
    /// 64-bit 浮動小数点。
    Float64,
    /// 32-bit 符号付き整数。
    Int32,
    /// 64-bit 符号付き整数。
    Int64,
    /// 8-bit 符号付き整数。
    Int8,
    /// 8-bit 符号なし整数。
    Uint8,
}

impl PthDtype {
    /// 1 要素あたりのバイト数。
    #[must_use]
    pub const fn element_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::BFloat16 | Self::Float16 => 2,
            Self::Float64 | Self::Int64 => 8,
            Self::Int8 | Self::Uint8 => 1,
        }
    }
}

/// テンソルのメタデータ。
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// データ型。
    pub dtype: PthDtype,
    /// 形状。
    pub shape: Vec<usize>,
    /// ZIP 内のストレージキー (`data/<key>`)。
    pub storage_key: String,
    /// ストレージ内のバイトオフセット。
    pub storage_offset: usize,
}

impl TensorMeta {
    /// テンソルの全要素数。
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// ストレージ内の読み込み開始バイト位置。
    #[must_use]
    pub fn byte_offset(self: &TensorMeta) -> usize {
        self.storage_offset * self.dtype.element_size()
    }

    /// テンソルデータのバイト数。
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.element_size()
    }
}

/// `.pth` ファイルローダー。
pub struct PthLoader {
    /// ZIP アーカイブのバイト列（全体をメモリに保持）。
    data: Vec<u8>,
    /// テンソル名 → メタデータ。
    tensors: HashMap<String, TensorMeta>,
}

impl PthLoader {
    /// `.pth` ファイルを開く。
    ///
    /// # Errors
    ///
    /// - ファイル読み込み失敗
    /// - ZIP パース失敗
    /// - pickle メタデータ解析失敗
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(data)
    }

    /// バイト列から構築する（テスト用）。
    ///
    /// # Errors
    ///
    /// ZIP / pickle パースエラー。
    pub fn from_bytes(data: Vec<u8>) -> io::Result<Self> {
        let tensors = parse_pth(&data)?;
        Ok(Self { data, tensors })
    }

    /// ローダーが保持するテンソル数。
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// テンソル名の一覧（順序不定）。
    #[must_use]
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// テンソルのメタデータを取得する。
    #[must_use]
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.tensors.get(name)
    }

    /// テンソルを FP32 Vec として取得する。
    ///
    /// BF16 / F16 / F64 / Int 系は FP32 に変換する。
    /// テンソルが存在しない場合は `None`。
    #[must_use]
    pub fn get_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        let meta = self.tensors.get(name)?;
        let storage_key = format!("archive/data/{}", meta.storage_key);
        let raw = read_zip_entry(&self.data, &storage_key)?;
        let offset = meta.byte_offset();
        let size = meta.byte_size();
        let slice = raw.get(offset..offset + size)?;
        Some(convert_to_f32(slice, meta.dtype, meta.numel()))
    }

    /// テンソルを生バイト列として取得する（dtype 変換なし）。
    ///
    /// 戻り値: `(bytes, dtype, shape)`。
    #[must_use]
    pub fn get_tensor_raw(&self, name: &str) -> Option<(Vec<u8>, PthDtype, Vec<usize>)> {
        let meta = self.tensors.get(name)?;
        let storage_key = format!("archive/data/{}", meta.storage_key);
        let raw = read_zip_entry(&self.data, &storage_key)?;
        let offset = meta.byte_offset();
        let size = meta.byte_size();
        let slice = raw.get(offset..offset + size)?;
        Some((slice.to_vec(), meta.dtype, meta.shape.clone()))
    }
}

// ---------------------------------------------------------------------------
// ZIP エントリ読み込み
// ---------------------------------------------------------------------------

/// ZIP アーカイブからエントリのバイト列を取得する。
fn read_zip_entry(data: &[u8], entry_name: &str) -> Option<Vec<u8>> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).ok()?;
    let mut entry = archive.by_name(entry_name).ok()?;
    let mut buf = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut buf).ok()?;
    Some(buf)
}

// ---------------------------------------------------------------------------
// pickle パーサー（最小実装）
// ---------------------------------------------------------------------------

/// `.pth` ZIP を解析し、テンソルメタデータを抽出する。
fn parse_pth(data: &[u8]) -> io::Result<HashMap<String, TensorMeta>> {
    let cursor = std::io::Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // data.pkl を読み込む
    let pkl_bytes = {
        let mut entry = archive.by_name("archive/data.pkl").map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("archive/data.pkl が見つかりません: {e}"),
            )
        })?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        buf
    };

    parse_pickle(&pkl_bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("pickle パースエラー: {e}")))
}

/// pickle バイト列を解析してテンソルメタデータを抽出する。
///
/// PyTorch が出力する pickle の構造は固定パターンなので、
/// 完全な pickle VM ではなくパターンマッチで必要情報を抽出する。
fn parse_pickle(pkl: &[u8]) -> Result<HashMap<String, TensorMeta>, String> {
    let mut parser = PickleParser::new(pkl);
    parser.extract_state_dict()
}

// ---------------------------------------------------------------------------
// PickleParser
// ---------------------------------------------------------------------------

/// pickle オペコード（使用するものだけ定義）。
#[allow(dead_code)]
mod opcode {
    pub const PROTO: u8 = 0x80;
    pub const FRAME: u8 = 0x95;
    pub const MARK: u8 = b'(';
    pub const DICT: u8 = b'd';
    pub const SETITEMS: u8 = b'u';
    pub const SETITEM: u8 = b's';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    pub const DUP: u8 = b'2';
    pub const SHORT_BINUNICODE: u8 = 0x8C;
    pub const BINUNICODE: u8 = b'X';
    pub const BINUNICODE8: u8 = 0x8D;
    pub const EMPTY_DICT: u8 = b'}';
    pub const EMPTY_LIST: u8 = b']';
    pub const EMPTY_TUPLE: u8 = b')';
    pub const APPENDS: u8 = b'e';
    pub const APPEND: u8 = b'a';
    pub const TUPLE: u8 = b't';
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    pub const NEWOBJ: u8 = b'R'; // actually REDUCE
    pub const REDUCE: u8 = b'R';
    pub const GLOBAL: u8 = b'c';
    pub const INST: u8 = b'i';
    pub const OBJ: u8 = b'o';
    pub const BUILD: u8 = b'b';
    pub const NONE: u8 = b'N';
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const LONG1: u8 = 0x8A;
    pub const LONG4: u8 = 0x8B;
    pub const INT: u8 = b'I';
    pub const LONG: u8 = b'L';
    pub const FLOAT: u8 = b'F';
    pub const BINFLOAT: u8 = b'G';
    pub const STRING: u8 = b'S';
    pub const SHORT_BINSTRING: u8 = b'U';
    pub const BINSTRING: u8 = b'T';
    pub const BINBYTES: u8 = b'B';
    pub const SHORT_BINBYTES: u8 = b'C';
    pub const BINBYTES8: u8 = 0x8E;
    pub const STOP: u8 = b'.';
    pub const PUT: u8 = b'p';
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const GET: u8 = b'g';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    pub const NEWFALSE: u8 = 0x89;
    pub const NEWTRUE: u8 = 0x88;
    pub const FROZENSET: u8 = 0x91;
    pub const BYTEARRAY8: u8 = 0x96;
    pub const NEXT_BUFFER: u8 = 0x97;
    pub const READONLY_BUFFER: u8 = 0x98;
}

/// pickle スタック上の値。
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum Value {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    Dict(Vec<(Value, Value)>),
    /// `GLOBAL` または `INST` で生成されたオブジェクト参照。
    Global { module: String, name: String },
    /// `REDUCE` / `NEWOBJ` で生成された呼び出し結果。
    /// PyTorch のテンソル storage は (global, (storage_key, location, dtype_str, numel)) の形式。
    Reduced { func: Box<Value>, args: Box<Value> },
    /// `BUILD` で付加された state を持つオブジェクト。
    Built { obj: Box<Value>, state: Box<Value> },
    /// マーク。
    Mark,
}

struct PickleParser<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

impl<'a> PickleParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        }
    }

    fn read_byte(&mut self) -> Result<u8, String> {
        if self.pos >= self.data.len() {
            return Err("データが終端に達しました".into());
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&[u8], String> {
        if self.pos + n > self.data.len() {
            return Err(format!("データ不足: pos={} n={} len={}", self.pos, n, self.data.len()));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        self.read_byte()
    }

    fn read_u16_le(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32_le(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32_le(&mut self) -> Result<i32, String> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64_le(&mut self) -> Result<u64, String> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_line(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        let line = std::str::from_utf8(&self.data[start..self.pos])
            .map_err(|e| format!("UTF-8エラー: {e}"))?
            .to_string();
        if self.pos < self.data.len() {
            self.pos += 1; // skip '\n'
        }
        Ok(line)
    }

    fn pop(&mut self) -> Result<Value, String> {
        self.stack.pop().ok_or_else(|| "スタックが空です".to_string())
    }

    fn peek(&self) -> Result<&Value, String> {
        self.stack.last().ok_or_else(|| "スタックが空です".to_string())
    }

    /// MARK までのスタック要素を取り出す。
    fn pop_mark(&mut self) -> Result<Vec<Value>, String> {
        let mut items = Vec::new();
        loop {
            let v = self.pop()?;
            if matches!(v, Value::Mark) {
                break;
            }
            items.push(v);
        }
        items.reverse();
        Ok(items)
    }

    /// pickle を実行し、最終的なスタックトップを返す。
    fn run(&mut self) -> Result<Value, String> {
        loop {
            let op = self.read_byte()?;
            match op {
                opcode::PROTO => {
                    let _version = self.read_u8()?;
                }
                opcode::FRAME => {
                    let _frame_len = self.read_u64_le()?;
                }
                opcode::MARK => {
                    self.stack.push(Value::Mark);
                }
                opcode::EMPTY_DICT => {
                    self.stack.push(Value::Dict(Vec::new()));
                }
                opcode::EMPTY_LIST => {
                    self.stack.push(Value::List(Vec::new()));
                }
                opcode::EMPTY_TUPLE => {
                    self.stack.push(Value::Tuple(Vec::new()));
                }
                opcode::DICT => {
                    let items = self.pop_mark()?;
                    let mut pairs = Vec::new();
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        pairs.push((k, v));
                    }
                    self.stack.push(Value::Dict(pairs));
                }
                opcode::SETITEM => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    if let Some(Value::Dict(ref mut d)) = self.stack.last_mut() {
                        d.push((k, v));
                    }
                }
                opcode::SETITEMS => {
                    let items = self.pop_mark()?;
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        if let Some(Value::Dict(ref mut d)) = self.stack.last_mut() {
                            d.push((k, v));
                        }
                    }
                }
                opcode::APPEND => {
                    let v = self.pop()?;
                    if let Some(Value::List(ref mut l)) = self.stack.last_mut() {
                        l.push(v);
                    }
                }
                opcode::APPENDS => {
                    let items = self.pop_mark()?;
                    if let Some(Value::List(ref mut l)) = self.stack.last_mut() {
                        l.extend(items);
                    }
                }
                opcode::TUPLE => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                opcode::TUPLE1 => {
                    let v = self.pop()?;
                    self.stack.push(Value::Tuple(vec![v]));
                }
                opcode::TUPLE2 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b]));
                }
                opcode::TUPLE3 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(vec![a, b, c]));
                }
                opcode::NONE => {
                    self.stack.push(Value::None);
                }
                opcode::NEWTRUE => {
                    self.stack.push(Value::Bool(true));
                }
                opcode::NEWFALSE => {
                    self.stack.push(Value::Bool(false));
                }
                opcode::BININT1 => {
                    let v = self.read_u8()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                opcode::BININT2 => {
                    let v = self.read_u16_le()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                opcode::BININT => {
                    let v = self.read_i32_le()?;
                    self.stack.push(Value::Int(i64::from(v)));
                }
                opcode::LONG1 => {
                    let n = self.read_u8()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let val = long_bytes_to_i64(bytes);
                    self.stack.push(Value::Int(val));
                }
                opcode::LONG4 => {
                    let n = self.read_u32_le()? as usize;
                    let bytes = self.read_bytes(n)?;
                    let val = long_bytes_to_i64(bytes);
                    self.stack.push(Value::Int(val));
                }
                opcode::INT => {
                    let s = self.read_line()?;
                    let v: i64 = s.trim().parse().unwrap_or(0);
                    self.stack.push(Value::Int(v));
                }
                opcode::LONG => {
                    let s = self.read_line()?;
                    let s = s.trim().trim_end_matches('L');
                    let v: i64 = s.parse().unwrap_or(0);
                    self.stack.push(Value::Int(v));
                }
                opcode::FLOAT => {
                    let s = self.read_line()?;
                    let v: f64 = s.trim().parse().unwrap_or(0.0);
                    self.stack.push(Value::Float(v));
                }
                opcode::BINFLOAT => {
                    let b = self.read_bytes(8)?;
                    let bits = u64::from_be_bytes(b.try_into().unwrap());
                    self.stack.push(Value::Float(f64::from_bits(bits)));
                }
                opcode::SHORT_BINUNICODE => {
                    let n = self.read_u8()? as usize;
                    let b = self.read_bytes(n)?;
                    let s = std::str::from_utf8(b).unwrap_or("").to_string();
                    self.stack.push(Value::Str(s));
                }
                opcode::BINUNICODE => {
                    let n = self.read_u32_le()? as usize;
                    let b = self.read_bytes(n)?;
                    let s = std::str::from_utf8(b).unwrap_or("").to_string();
                    self.stack.push(Value::Str(s));
                }
                opcode::BINUNICODE8 => {
                    let n = self.read_u64_le()? as usize;
                    let b = self.read_bytes(n)?;
                    let s = std::str::from_utf8(b).unwrap_or("").to_string();
                    self.stack.push(Value::Str(s));
                }
                opcode::STRING => {
                    let s = self.read_line()?;
                    // pickle STRING: 'foo' または "foo"
                    let s = s.trim();
                    let s = if (s.starts_with('\'') && s.ends_with('\''))
                        || (s.starts_with('"') && s.ends_with('"'))
                    {
                        s[1..s.len() - 1].to_string()
                    } else {
                        s.to_string()
                    };
                    self.stack.push(Value::Str(s));
                }
                opcode::SHORT_BINSTRING => {
                    let n = self.read_u8()? as usize;
                    let b = self.read_bytes(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                opcode::BINSTRING => {
                    let n = self.read_i32_le()? as usize;
                    let b = self.read_bytes(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                opcode::SHORT_BINBYTES => {
                    let n = self.read_u8()? as usize;
                    let b = self.read_bytes(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                opcode::BINBYTES => {
                    let n = self.read_u32_le()? as usize;
                    let b = self.read_bytes(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                opcode::BINBYTES8 => {
                    let n = self.read_u64_le()? as usize;
                    let b = self.read_bytes(n)?.to_vec();
                    self.stack.push(Value::Bytes(b));
                }
                opcode::GLOBAL => {
                    let module = self.read_line()?;
                    let name = self.read_line()?;
                    self.stack.push(Value::Global {
                        module: module.trim().to_string(),
                        name: name.trim().to_string(),
                    });
                }
                opcode::REDUCE => {
                    let args = self.pop()?;
                    let func = self.pop()?;
                    self.stack.push(Value::Reduced {
                        func: Box::new(func),
                        args: Box::new(args),
                    });
                }
                opcode::BUILD => {
                    let state = self.pop()?;
                    let obj = self.pop()?;
                    self.stack.push(Value::Built {
                        obj: Box::new(obj),
                        state: Box::new(state),
                    });
                }
                opcode::BINPUT => {
                    let idx = self.read_u8()? as u32;
                    let v = self.peek()?.clone();
                    self.memo.insert(idx, v);
                }
                opcode::LONG_BINPUT => {
                    let idx = self.read_u32_le()?;
                    let v = self.peek()?.clone();
                    self.memo.insert(idx, v);
                }
                opcode::PUT => {
                    let s = self.read_line()?;
                    let idx: u32 = s.trim().parse().unwrap_or(0);
                    let v = self.peek()?.clone();
                    self.memo.insert(idx, v);
                }
                opcode::BINGET => {
                    let idx = self.read_u8()? as u32;
                    let v = self.memo.get(&idx).cloned().ok_or_else(|| {
                        format!("memo[{idx}] が見つかりません")
                    })?;
                    self.stack.push(v);
                }
                opcode::LONG_BINGET => {
                    let idx = self.read_u32_le()?;
                    let v = self.memo.get(&idx).cloned().ok_or_else(|| {
                        format!("memo[{idx}] が見つかりません")
                    })?;
                    self.stack.push(v);
                }
                opcode::GET => {
                    let s = self.read_line()?;
                    let idx: u32 = s.trim().parse().unwrap_or(0);
                    let v = self.memo.get(&idx).cloned().ok_or_else(|| {
                        format!("memo[{idx}] が見つかりません")
                    })?;
                    self.stack.push(v);
                }
                opcode::POP => {
                    let _ = self.pop()?;
                }
                opcode::POP_MARK => {
                    let _ = self.pop_mark()?;
                }
                opcode::DUP => {
                    let v = self.peek()?.clone();
                    self.stack.push(v);
                }
                opcode::STOP => {
                    return self.pop();
                }
                opcode::FROZENSET => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(items));
                }
                opcode::BYTEARRAY8 | opcode::NEXT_BUFFER | opcode::READONLY_BUFFER => {
                    // 未対応 opcode: 無視
                }
                other => {
                    return Err(format!(
                        "未対応 opcode: 0x{other:02X} (pos={})",
                        self.pos - 1
                    ));
                }
            }
        }
    }

    /// pickle を実行し、state_dict を抽出する。
    fn extract_state_dict(&mut self) -> Result<HashMap<String, TensorMeta>, String> {
        let root = self.run()?;
        let mut result = HashMap::new();
        extract_tensors_from_value(&root, "", &mut result);
        Ok(result)
    }
}

/// pickle の LONG バイト列 (little-endian 2's complement) → i64。
fn long_bytes_to_i64(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let negative = bytes.last().map_or(false, |&b| b & 0x80 != 0);
    let mut val: i64 = 0;
    for (i, &b) in bytes.iter().enumerate().take(8) {
        val |= (i64::from(b)) << (i * 8);
    }
    if negative && bytes.len() < 8 {
        // 符号拡張
        let shift = bytes.len() * 8;
        val |= !((1i64 << shift) - 1);
    }
    val
}

// ---------------------------------------------------------------------------
// 値ツリーからテンソルメタデータを抽出
// ---------------------------------------------------------------------------

/// `Value` ツリーを再帰的に走査し、テンソルメタデータを収集する。
fn extract_tensors_from_value(val: &Value, prefix: &str, result: &mut HashMap<String, TensorMeta>) {
    match val {
        Value::Dict(pairs) => {
            for (k, v) in pairs {
                let key = match k {
                    Value::Str(s) => s.clone(),
                    Value::Int(i) => i.to_string(),
                    _ => continue,
                };
                let full = if prefix.is_empty() {
                    key
                } else {
                    format!("{prefix}.{key}")
                };
                // テンソルかどうか試みる
                if let Some(meta) = try_extract_tensor(v) {
                    result.insert(full, meta);
                } else {
                    // ネスト辞書の場合は再帰
                    extract_tensors_from_value(v, &full, result);
                }
            }
        }
        Value::Built { obj, state } => {
            // OrderedDict などの Built オブジェクトの state を探索
            extract_tensors_from_value(obj, prefix, result);
            extract_tensors_from_value(state, prefix, result);
        }
        Value::Reduced { func: _, args } => {
            // REDUCE の結果自体がテンソルの場合は try_extract_tensor で処理済み
            // args がリストの場合は再帰
            if let Value::Tuple(items) | Value::List(items) = args.as_ref() {
                for (i, item) in items.iter().enumerate() {
                    let key = if prefix.is_empty() {
                        i.to_string()
                    } else {
                        format!("{prefix}.{i}")
                    };
                    extract_tensors_from_value(item, &key, result);
                }
            }
        }
        _ => {}
    }
}

/// `Value` がテンソルの場合、`TensorMeta` を返す。
///
/// PyTorch が pickle に書き出すテンソルの構造:
/// ```text
/// REDUCE(
///   func = GLOBAL("torch._utils", "_rebuild_tensor_v2"),
///   args = TUPLE(
///     REDUCE(storage, (storage_key, location, numel)),  // storage
///     storage_offset,
///     shape_tuple,
///     stride_tuple,
///     requires_grad,
///     backward_hooks,
///   )
/// )
/// ```
fn try_extract_tensor(val: &Value) -> Option<TensorMeta> {
    let (func, args) = match val {
        Value::Reduced { func, args } => (func.as_ref(), args.as_ref()),
        _ => return None,
    };

    // func が _rebuild_tensor_v2 か確認
    let is_rebuild = match func {
        Value::Global { module, name } => {
            module == "torch._utils" && name == "_rebuild_tensor_v2"
        }
        _ => false,
    };
    if !is_rebuild {
        return None;
    }

    let args = match args {
        Value::Tuple(v) => v,
        _ => return None,
    };
    if args.len() < 3 {
        return None;
    }

    // args[0] = storage (REDUCE of storage type)
    let (storage_key, dtype) = extract_storage_info(&args[0])?;

    // args[1] = storage_offset
    let storage_offset = match &args[1] {
        Value::Int(i) => *i as usize,
        _ => 0,
    };

    // args[2] = shape tuple
    let shape = extract_shape(&args[2])?;

    Some(TensorMeta {
        dtype,
        shape,
        storage_key,
        storage_offset,
    })
}

/// storage の REDUCE から (storage_key, dtype) を抽出する。
fn extract_storage_info(val: &Value) -> Option<(String, PthDtype)> {
    let (func, args) = match val {
        Value::Reduced { func, args } => (func.as_ref(), args.as_ref()),
        _ => return None,
    };

    // func が storage type global か確認し dtype を取得
    let dtype = match func {
        Value::Global { module, name } => {
            if module != "torch" && module != "torch.storage" {
                return None;
            }
            dtype_from_storage_class(name)?
        }
        _ => return None,
    };

    // args = TUPLE(storage_key, location, numel) または
    //        TUPLE(storage_key, location, device, numel) など
    let args = match args {
        Value::Tuple(v) => v,
        _ => return None,
    };
    if args.is_empty() {
        return None;
    }

    let storage_key = match &args[0] {
        Value::Str(s) => s.clone(),
        _ => return None,
    };

    Some((storage_key, dtype))
}

/// PyTorch ストレージクラス名 → `PthDtype`。
fn dtype_from_storage_class(name: &str) -> Option<PthDtype> {
    match name {
        "FloatStorage" | "float32" | "_UntypedStorage" => Some(PthDtype::Float32),
        "BFloat16Storage" | "bfloat16" => Some(PthDtype::BFloat16),
        "HalfStorage" | "float16" | "half" => Some(PthDtype::Float16),
        "DoubleStorage" | "float64" | "double" => Some(PthDtype::Float64),
        "IntStorage" | "int32" | "int" => Some(PthDtype::Int32),
        "LongStorage" | "int64" | "long" => Some(PthDtype::Int64),
        "CharStorage" | "int8" | "char" => Some(PthDtype::Int8),
        "ByteStorage" | "uint8" | "byte" => Some(PthDtype::Uint8),
        _ => {
            // 未知の dtype は Float32 として扱う（_UntypedStorage 等）
            Some(PthDtype::Float32)
        }
    }
}

/// shape タプルを `Vec<usize>` に変換する。
fn extract_shape(val: &Value) -> Option<Vec<usize>> {
    match val {
        Value::Tuple(items) | Value::List(items) => {
            let mut shape = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Value::Int(i) => shape.push(*i as usize),
                    _ => return None,
                }
            }
            Some(shape)
        }
        Value::Int(i) => Some(vec![*i as usize]),
        Value::None => Some(vec![]), // scalar
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// dtype 変換
// ---------------------------------------------------------------------------

/// バイト列を FP32 Vec に変換する。
fn convert_to_f32(data: &[u8], dtype: PthDtype, numel: usize) -> Vec<f32> {
    match dtype {
        PthDtype::Float32 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 4;
                if base + 4 <= data.len() {
                    out.push(f32::from_le_bytes([
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                    ]));
                }
            }
            out
        }
        PthDtype::BFloat16 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 2;
                if base + 2 <= data.len() {
                    let bits = u16::from_le_bytes([data[base], data[base + 1]]);
                    out.push(f32::from_bits(u32::from(bits) << 16));
                }
            }
            out
        }
        PthDtype::Float16 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 2;
                if base + 2 <= data.len() {
                    let bits = u16::from_le_bytes([data[base], data[base + 1]]);
                    out.push(f16_bits_to_f32(bits));
                }
            }
            out
        }
        PthDtype::Float64 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 8;
                if base + 8 <= data.len() {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[base..base + 8]);
                    out.push(f64::from_le_bytes(bytes) as f32);
                }
            }
            out
        }
        PthDtype::Int32 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 4;
                if base + 4 <= data.len() {
                    let v = i32::from_le_bytes([
                        data[base],
                        data[base + 1],
                        data[base + 2],
                        data[base + 3],
                    ]);
                    out.push(v as f32);
                }
            }
            out
        }
        PthDtype::Int64 => {
            let mut out = Vec::with_capacity(numel);
            for i in 0..numel {
                let base = i * 8;
                if base + 8 <= data.len() {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&data[base..base + 8]);
                    out.push(i64::from_le_bytes(bytes) as f32);
                }
            }
            out
        }
        PthDtype::Int8 => data.iter().take(numel).map(|&b| b as i8 as f32).collect(),
        PthDtype::Uint8 => data.iter().take(numel).map(|&b| b as f32).collect(),
    }
}

/// IEEE 754 half-precision ビット列 → f32。
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15);
    let exp = u32::from((h >> 10) & 0x1F);
    let mant = u32::from(h & 0x3FF);
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let exp32 = (127u32 - 15 + 1).wrapping_add(e as u32);
        return f32::from_bits((sign << 31) | (exp32 << 23) | (m << 13));
    }
    if exp == 31 {
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits((sign << 31) | (exp32 << 23) | (mant << 13))
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // dtype 変換テスト
    // -----------------------------------------------------------------------

    #[test]
    fn convert_f32_roundtrip() {
        let values = [1.0f32, -2.5, 0.0, 3.14];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = convert_to_f32(&data, PthDtype::Float32, values.len());
        assert_eq!(result.len(), values.len());
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1e-6, "期待値 {b}、実際値 {a}");
        }
    }

    #[test]
    fn convert_bf16_one() {
        // 1.0 in BF16 = 0x3F80
        let data = [0x80u8, 0x3F];
        let result = convert_to_f32(&data, PthDtype::BFloat16, 1);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn convert_bf16_negative() {
        // -1.0 in BF16 = 0xBF80
        let data = [0x80u8, 0xBF];
        let result = convert_to_f32(&data, PthDtype::BFloat16, 1);
        assert!((result[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn convert_f16_one() {
        // 1.0 in F16 = 0x3C00
        let data = [0x00u8, 0x3C];
        let result = convert_to_f32(&data, PthDtype::Float16, 1);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn convert_f16_zero() {
        let data = [0x00u8, 0x00];
        let result = convert_to_f32(&data, PthDtype::Float16, 1);
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn convert_f16_minus_two() {
        // -2.0 in F16 = 0xC000
        let data = [0x00u8, 0xC0];
        let result = convert_to_f32(&data, PthDtype::Float16, 1);
        assert!((result[0] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn convert_f64_value() {
        let v: f64 = 2.718_281_828_459_045;
        let mut data = Vec::new();
        data.extend_from_slice(&v.to_le_bytes());
        let result = convert_to_f32(&data, PthDtype::Float64, 1);
        assert!((result[0] - v as f32).abs() < 1e-5);
    }

    #[test]
    fn convert_int32_values() {
        let values = [0i32, 1, -1, 100, -100];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = convert_to_f32(&data, PthDtype::Int32, values.len());
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((*a - *b as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn convert_int64_values() {
        let values = [0i64, 1, -1, 1000];
        let mut data = Vec::new();
        for &v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let result = convert_to_f32(&data, PthDtype::Int64, values.len());
        for (a, b) in result.iter().zip(values.iter()) {
            assert!((*a - *b as f32).abs() < 1e-6);
        }
    }

    #[test]
    fn convert_int8_values() {
        let data = [0u8, 127, 128, 255]; // 0, 127, -128, -1 as i8
        let result = convert_to_f32(&data, PthDtype::Int8, 4);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 127.0);
        assert_eq!(result[2], -128.0);
        assert_eq!(result[3], -1.0);
    }

    #[test]
    fn convert_uint8_values() {
        let data = [0u8, 1, 128, 255];
        let result = convert_to_f32(&data, PthDtype::Uint8, 4);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 128.0);
        assert_eq!(result[3], 255.0);
    }

    // -----------------------------------------------------------------------
    // dtype サイズテスト
    // -----------------------------------------------------------------------

    #[test]
    fn dtype_element_size() {
        assert_eq!(PthDtype::Float32.element_size(), 4);
        assert_eq!(PthDtype::BFloat16.element_size(), 2);
        assert_eq!(PthDtype::Float16.element_size(), 2);
        assert_eq!(PthDtype::Float64.element_size(), 8);
        assert_eq!(PthDtype::Int32.element_size(), 4);
        assert_eq!(PthDtype::Int64.element_size(), 8);
        assert_eq!(PthDtype::Int8.element_size(), 1);
        assert_eq!(PthDtype::Uint8.element_size(), 1);
    }

    // -----------------------------------------------------------------------
    // TensorMeta テスト
    // -----------------------------------------------------------------------

    #[test]
    fn tensor_meta_numel_1d() {
        let meta = TensorMeta {
            dtype: PthDtype::Float32,
            shape: vec![256],
            storage_key: "0".to_string(),
            storage_offset: 0,
        };
        assert_eq!(meta.numel(), 256);
        assert_eq!(meta.byte_size(), 256 * 4);
    }

    #[test]
    fn tensor_meta_numel_2d() {
        let meta = TensorMeta {
            dtype: PthDtype::BFloat16,
            shape: vec![4, 8],
            storage_key: "0".to_string(),
            storage_offset: 0,
        };
        assert_eq!(meta.numel(), 32);
        assert_eq!(meta.byte_size(), 32 * 2);
    }

    #[test]
    fn tensor_meta_numel_scalar() {
        let meta = TensorMeta {
            dtype: PthDtype::Float32,
            shape: vec![],
            storage_key: "0".to_string(),
            storage_offset: 0,
        };
        assert_eq!(meta.numel(), 1);
    }

    #[test]
    fn tensor_meta_byte_offset() {
        let meta = TensorMeta {
            dtype: PthDtype::Float32,
            shape: vec![10],
            storage_key: "0".to_string(),
            storage_offset: 5,
        };
        assert_eq!(meta.byte_offset(), 5 * 4);
    }

    // -----------------------------------------------------------------------
    // pickle パーサーテスト
    // -----------------------------------------------------------------------

    #[test]
    fn pickle_parse_empty_dict() {
        // PROTO 2, EMPTY_DICT, STOP
        let pkl = [0x80, 2, b'}', b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run();
        assert!(result.is_ok(), "パース失敗: {:?}", result.err());
        matches!(result.unwrap(), Value::Dict(_));
    }

    #[test]
    fn pickle_parse_binint1() {
        // PROTO 2, BININT1(42), STOP
        let pkl = [0x80, 2, b'K', 42, b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::Int(42)));
    }

    #[test]
    fn pickle_parse_binint2() {
        // PROTO 2, BININT2(1000), STOP
        let pkl = [0x80, 2, b'M', 0xE8, 0x03, b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::Int(1000)));
    }

    #[test]
    fn pickle_parse_short_binunicode() {
        // PROTO 2, SHORT_BINUNICODE("ab"), STOP
        let pkl = [0x80, 2, 0x8C, 2, b'a', b'b', b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::Str(ref s) if s == "ab"));
    }

    #[test]
    fn pickle_parse_tuple2() {
        // PROTO 2, BININT1(1), BININT1(2), TUPLE2, STOP
        let pkl = [0x80, 2, b'K', 1, b'K', 2, 0x86, b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::Tuple(ref v) if v.len() == 2));
    }

    #[test]
    fn pickle_parse_none() {
        let pkl = [0x80, 2, b'N', b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::None));
    }

    #[test]
    fn pickle_parse_bool() {
        let pkl_true = [0x80, 2, 0x88u8, b'.'];
        let pkl_false = [0x80, 2, 0x89u8, b'.'];
        let mut p = PickleParser::new(&pkl_true);
        assert!(matches!(p.run().unwrap(), Value::Bool(true)));
        let mut p = PickleParser::new(&pkl_false);
        assert!(matches!(p.run().unwrap(), Value::Bool(false)));
    }

    #[test]
    fn pickle_parse_memo_put_get() {
        // PROTO 2, BININT1(99), BINPUT(0), POP, BINGET(0), STOP
        let pkl = [0x80, 2, b'K', 99, b'q', 0, b'0', b'h', 0, b'.'];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        assert!(matches!(result, Value::Int(99)));
    }

    #[test]
    fn pickle_parse_dict_setitem() {
        // PROTO 2, EMPTY_DICT, MARK, str("a"), int(1), SETITEMS, STOP
        let pkl = [
            0x80, 2,
            b'}',         // EMPTY_DICT
            b'(',         // MARK
            0x8C, 1, b'a', // SHORT_BINUNICODE "a"
            b'K', 1,      // BININT1(1)
            b'u',         // SETITEMS
            b'.',         // STOP
        ];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        if let Value::Dict(pairs) = result {
            assert_eq!(pairs.len(), 1);
            assert!(matches!(&pairs[0].0, Value::Str(s) if s == "a"));
            assert!(matches!(pairs[0].1, Value::Int(1)));
        } else {
            panic!("Dict ではありません: {result:?}");
        }
    }

    #[test]
    fn pickle_parse_list_append() {
        // PROTO 2, EMPTY_LIST, MARK, int(1), int(2), APPENDS, STOP
        let pkl = [
            0x80, 2,
            b']',         // EMPTY_LIST
            b'(',         // MARK
            b'K', 1,      // BININT1(1)
            b'K', 2,      // BININT1(2)
            b'e',         // APPENDS
            b'.',         // STOP
        ];
        let mut parser = PickleParser::new(&pkl);
        let result = parser.run().unwrap();
        if let Value::List(items) = result {
            assert_eq!(items.len(), 2);
        } else {
            panic!("List ではありません");
        }
    }

    // -----------------------------------------------------------------------
    // dtype_from_storage_class テスト
    // -----------------------------------------------------------------------

    #[test]
    fn storage_class_to_dtype() {
        assert_eq!(dtype_from_storage_class("FloatStorage"), Some(PthDtype::Float32));
        assert_eq!(dtype_from_storage_class("BFloat16Storage"), Some(PthDtype::BFloat16));
        assert_eq!(dtype_from_storage_class("HalfStorage"), Some(PthDtype::Float16));
        assert_eq!(dtype_from_storage_class("DoubleStorage"), Some(PthDtype::Float64));
        assert_eq!(dtype_from_storage_class("LongStorage"), Some(PthDtype::Int64));
        assert_eq!(dtype_from_storage_class("IntStorage"), Some(PthDtype::Int32));
        assert_eq!(dtype_from_storage_class("CharStorage"), Some(PthDtype::Int8));
        assert_eq!(dtype_from_storage_class("ByteStorage"), Some(PthDtype::Uint8));
    }

    // -----------------------------------------------------------------------
    // extract_shape テスト
    // -----------------------------------------------------------------------

    #[test]
    fn extract_shape_from_tuple() {
        let v = Value::Tuple(vec![Value::Int(3), Value::Int(4)]);
        let shape = extract_shape(&v).unwrap();
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn extract_shape_scalar() {
        let v = Value::None;
        let shape = extract_shape(&v).unwrap();
        assert!(shape.is_empty());
    }

    // -----------------------------------------------------------------------
    // f16_bits_to_f32 テスト
    // -----------------------------------------------------------------------

    #[test]
    fn f16_bits_one() {
        assert!((f16_bits_to_f32(0x3C00) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn f16_bits_zero() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
    }

    #[test]
    fn f16_bits_minus_two() {
        assert!((f16_bits_to_f32(0xC000) + 2.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // long_bytes_to_i64 テスト
    // -----------------------------------------------------------------------

    #[test]
    fn long_bytes_zero() {
        assert_eq!(long_bytes_to_i64(&[]), 0);
    }

    #[test]
    fn long_bytes_positive() {
        // 42 = 0x2A
        assert_eq!(long_bytes_to_i64(&[0x2A]), 42);
    }

    #[test]
    fn long_bytes_negative() {
        // -1 in 1-byte 2's complement = 0xFF
        assert_eq!(long_bytes_to_i64(&[0xFF]), -1);
    }

    #[test]
    fn long_bytes_256() {
        // 256 = 0x00 0x01 (little-endian)
        assert_eq!(long_bytes_to_i64(&[0x00, 0x01]), 256);
    }
}
